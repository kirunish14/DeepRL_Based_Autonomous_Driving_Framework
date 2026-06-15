import gymnasium as gym
from gymnasium import spaces

import carla
import random
import numpy as np
import cv2
import time


class CarlaEnv(gym.Env):

    def __init__(self):

        super(CarlaEnv, self).__init__()

        # =========================
        # CONNECT TO CARLA
        # =========================

        self.client = carla.Client(
            'localhost',
            2000
        )

        self.client.set_timeout(10.0)

        self.world = self.client.load_world("Town10HD")

        print("Current Map:", self.world.get_map().name)

        print(
            "Current Map:",
            self.world.get_map().name
        )

        self.blueprint_library = (
            self.world.get_blueprint_library()
        )


        # =========================
        # ACTION SPACE
        # =========================

        # [steering, throttle]

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # =========================
        # OBSERVATION SPACE
        # =========================

        self.observation_space = spaces.Box(
            low=np.array([0, -20, -1, -1], dtype=np.float32),
            high=np.array([20, 20, 1, 1], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # =========================
        # VARIABLES
        # =========================

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None

        self.last_steer = 0
        self.previous_steer = 0
        self.previous_throttle = 0

        self.front_camera = None

        self.stuck_time = 0
        self.step_count = 0
        self.collision_hist = []
        self.lane_invasion_hist = []

        self.previous_location = None

        self.actor_list = []

    # =========================
    # RESET ENVIRONMENT
    # =========================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Destroy previous actors
        self.destroy_actors()
        self.stuck_time = 0
        self.step_count = 0

        time.sleep(1)

        # Reset histories
        self.collision_hist = []
        self.lane_invasion_hist = []

        self.previous_steer = 0
        self.last_steer = 0
        self.previous_throttle = 0

        self.front_camera = None

        # =========================
        # SPAWN VEHICLE
        # =========================

        vehicle_bp = (
            self.blueprint_library.filter(
                'model3'
            )[0]
        )

        spawn_points = (
            self.world.get_map().get_spawn_points()
        )

        for i, sp in enumerate(spawn_points):

            print(
                i,
                sp.location
            )

        self.vehicle = None

        spawn_point = spawn_points[1]

        for _ in range(20):

            self.vehicle = self.world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )

            if self.vehicle is not None:
                break

            time.sleep(0.5)

        if self.vehicle is None:
            raise RuntimeError(
                "Could not spawn vehicle"
            )

        print(f"Spawn Location: {spawn_point.location}")

        self.actor_list.append(
            self.vehicle
        )

        self.previous_location = (
            self.vehicle.get_location()
        )

        self.start_location = self.vehicle.get_location()

        self.previous_distance_from_start = 0

        # =========================
        # COLLISION SENSOR
        # =========================

        collision_bp = self.blueprint_library.find(
            'sensor.other.collision'
        )

        self.collision_sensor = self.world.try_spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )

        self.actor_list.append(
            self.collision_sensor
        )

        self.collision_sensor.listen(
            lambda event:
            self.collision_hist.append(event)
        )

        # =========================
        # LANE INVASION SENSOR
        # =========================

        lane_sensor_bp = self.blueprint_library.find(
            'sensor.other.lane_invasion'
        )

        self.lane_sensor = self.world.try_spawn_actor(
            lane_sensor_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )

        self.actor_list.append(
            self.lane_sensor
        )

        self.lane_sensor.listen(
            lambda event:
            self.lane_invasion_hist.append(event)
        )

        # =========================
        # CAMERA SENSOR
        # =========================

        camera_bp = self.blueprint_library.find(
            'sensor.camera.rgb'
        )

        camera_bp.set_attribute(
            'image_size_x',
            '160'
        )

        camera_bp.set_attribute(
            'image_size_y',
            '80'
        )

        camera_bp.set_attribute(
            'fov',
            '90'
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=2.5,
                z=1.5
            )
        )

        self.camera = self.world.try_spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        self.actor_list.append(
            self.camera
        )

        self.camera.listen(
            lambda data:
            self.process_image(data)
        )

        # Wait for camera image
        while self.front_camera is None:

            time.sleep(0.01)
        observation = np.array([
            0.0,
            0.0,
            1.0,
            1.0
        ], dtype=np.float32)

        info = {}

        return observation, info

    # =========================
    # PROCESS CAMERA IMAGE
    # =========================

    def process_image(self, image):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        array = np.reshape(
            array,
            (image.height, image.width, 4)
        )

        self.front_camera = array[:, :, :3]

    # =========================
    # STEP FUNCTION
    # =========================

    def step(self, action):
        

        done = False

        self.step_count += 1

        # End excessively long episodes
        if self.step_count > 3000:
            print("DONE: MAX STEPS")
            done = True

        # =========================
        # ACTION PROCESSING
        # =========================

        steer = np.clip(
            float(action[0]),
            -0.30,
            0.30
        )

        steer = (
            0.8 * self.previous_steer +
            0.2 * steer
        )

        self.previous_steer = steer


        throttle = float(action[1])

        throttle = np.clip(
            throttle,
            0.15,
            0.45
        )

        throttle = (
            0.8 * self.previous_throttle +
            0.2 * throttle
        )

        self.previous_throttle = throttle

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer
        )

        self.vehicle.apply_control(control)

        self.vehicle.apply_control(control)

        time.sleep(0.05)


        # =========================
        # REWARD
        # =========================

        reward = 0

        velocity = self.vehicle.get_velocity()

        speed = np.sqrt(
            velocity.x ** 2 +
            velocity.y ** 2
        )

        reward += speed * 0.05

        reward += 0.1

        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location()
        )
        
        distance_from_center = (
            self.vehicle.get_location().distance(
                waypoint.transform.location
            )
        )

        reward -= distance_from_center * 3

        vehicle_forward = (
            self.vehicle.get_transform().get_forward_vector()
        )

        road_forward = (
            waypoint.transform.get_forward_vector()
        )

        alignment = (
            vehicle_forward.x * road_forward.x +
            vehicle_forward.y * road_forward.y
        )

        next_waypoints = waypoint.next(5.0)

        if len(next_waypoints) > 0:

            next_wp = next_waypoints[0]

            next_forward = (
                next_wp.transform.get_forward_vector()
            )

            future_alignment = (
                vehicle_forward.x * next_forward.x +
                vehicle_forward.y * next_forward.y
            )

        else:

            future_alignment = alignment

        current_location = self.vehicle.get_location()

        distance_from_start = current_location.distance(
            self.start_location
        )

        progress = (
            distance_from_start -
            self.previous_distance_from_start
        )

        reward += progress * 2

        self.previous_distance_from_start = (
            distance_from_start
        )

        
        # =========================
        # STUCK DETECTION
        # =========================
        distance_travelled = current_location.distance(
            self.previous_location
        )

        if distance_travelled < 0.05:

            self.stuck_time += 1

        else:

            self.stuck_time = 0

        if self.stuck_time > 150:

            print("DONE: STUCK")

            reward -= 10

            done = True

        self.previous_location = current_location

        # =========================
        # COLLISION
        # =========================

        if len(self.collision_hist) != 0:

            print("DONE: COLLISION")

            reward -= 50

            done = True

        # =========================
        # LANE INVASION
        # =========================

        if len(self.lane_invasion_hist) != 0:

            reward -= 5

            self.lane_invasion_hist = []

        # =========================
        # STEERING PENALTY
        # =========================

        reward -= abs(steer) * 0.5
        reward -= abs(
            steer - self.last_steer
        ) * 0.3

        self.last_steer = steer

        # =========================
        # MAP BOUNDARY
        # =========================

        location = self.vehicle.get_location()

        if abs(location.x) > 500:
            
            print("DONE: OUT OF BOUNDS")

            done = True

        
        observation = np.array([
            speed,
            distance_from_center,
            alignment,
            future_alignment

        ], dtype=np.float32)

        if self.step_count % 100 == 0:
            print("State:", observation)

        info = {}
        if self.step_count % 50 == 0:

            print(
                f"Reward={reward:.2f} "
                f"Speed={speed:.2f} "
                f"Align={alignment:.2f} "
                f"Center={distance_from_center:.2f}"
                f"Steer={steer:.2f}"
            )

        return (
            observation,
            reward,
            done,
            False,
            info
        )

    # =========================
    # DESTROY ACTORS
    # =========================

    def destroy_actors(self):

        print("Destroying actors...")

        for actor in self.actor_list:

            try:

                if actor is not None:

                    actor.destroy()

            except Exception as e:

                print(
                    f"Error destroying actor: {e}"
                )

        self.actor_list = []

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None

        time.sleep(1)

    # =========================
    # CLOSE ENVIRONMENT
    # =========================

    def close(self):

        self.destroy_actors()

        self.world.tick()
        time.sleep(2)