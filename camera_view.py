import carla
import random
import cv2
import numpy as np
import time

# Global image variable
current_image = None

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

print("Vehicle Spawned")

# Camera setup
camera_bp = blueprint_library.find('sensor.camera.rgb')

camera_bp.set_attribute('image_size_x', '320')
camera_bp.set_attribute('image_size_y', '240')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(
    carla.Location(x=2.5, z=1.5)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

print("Camera Attached")

# Callback function
def process_image(image):
    global current_image

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3]

# Start listening
camera.listen(process_image)

try:
    while True:

        if current_image is not None:
            cv2.imshow("CARLA Camera", current_image)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    pass

finally:
    print("Cleaning up")

    camera.stop()

    camera.destroy()
    vehicle.destroy()

    cv2.destroyAllWindows()

    print("Done")
