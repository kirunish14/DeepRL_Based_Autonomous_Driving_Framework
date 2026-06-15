import carla
import random
import cv2
import numpy as np
import time

# =========================
# GLOBAL VARIABLES
# =========================

current_image = None

# PID Variables
previous_error = 0
integral = 0

# =========================
# CONNECT TO CARLA
# =========================

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()

blueprint_library = world.get_blueprint_library()

# =========================
# SPAWN VEHICLE
# =========================

vehicle_bp = blueprint_library.filter('model3')[0]

spawn_point = random.choice(
    world.get_map().get_spawn_points()
)

vehicle = world.spawn_actor(
    vehicle_bp,
    spawn_point
)

print("Tesla Spawned")

# =========================
# CAMERA SETUP
# =========================

camera_bp = blueprint_library.find(
    'sensor.camera.rgb'
)

camera_bp.set_attribute(
    'image_size_x',
    '640'
)

camera_bp.set_attribute(
    'image_size_y',
    '480'
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

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

print("Camera Attached")

# =========================
# IMAGE CALLBACK
# =========================

def process_image(image):

    global current_image

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    current_image = array[:, :, :3]

# Start camera
camera.listen(process_image)

# =========================
# REGION OF INTEREST
# =========================

def region_of_interest(img):

    height = img.shape[0]

    polygons = np.array([
        [
            (0, height),
            (640, height),
            (400, 250),
            (240, 250)
        ]
    ])

    mask = np.zeros_like(img)

    cv2.fillPoly(
        mask,
        polygons,
        255
    )

    masked_image = cv2.bitwise_and(
        img,
        mask
    )

    return masked_image

# =========================
# MAIN LOOP
# =========================

try:

    while True:

        if current_image is not None:

            # Copy frame
            frame = current_image.copy()

            # Convert to grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Gaussian blur
            blur = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            # Edge detection
            edges = cv2.Canny(
                blur,
                50,
                150
            )

            # ROI Mask
            cropped_edges = region_of_interest(
                edges
            )

            # Detect lane lines
            lines = cv2.HoughLinesP(
                cropped_edges,
                2,
                np.pi / 180,
                100,
                np.array([]),
                minLineLength=40,
                maxLineGap=5
            )

            # Empty image for drawing
            line_image = np.zeros_like(frame)

            left_x = []
            right_x = []

            # Draw detected lines
            if lines is not None:

                for line in lines:

                    x1, y1, x2, y2 = line.reshape(4)

                    cv2.line(
                        line_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        5
                    )

                    # Left lane
                    if x1 < 320 and x2 < 320:
                        left_x.extend([x1, x2])

                    # Right lane
                    elif x1 > 320 and x2 > 320:
                        right_x.extend([x1, x2])

            # Overlay lane lines
            combo = cv2.addWeighted(
                frame,
                0.8,
                line_image,
                1,
                1
            )

            # =========================
            # PID STEERING CONTROL
            # =========================

            steering = 0.0

            if left_x and right_x:

                # Calculate lane centers
                left_lane = int(np.mean(left_x))
                right_lane = int(np.mean(right_x))

                lane_center = (
                    left_lane + right_lane
                ) // 2

                frame_center = 320

                # Draw centers
                cv2.circle(
                    combo,
                    (lane_center, 400),
                    10,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    combo,
                    (frame_center, 400),
                    10,
                    (255, 0, 0),
                    -1
                )

                # Calculate error
                error = lane_center - frame_center

                # Normalize error
                error = error / 320

                # PID Constants
                Kp = 0.4
                Ki = 0.01
                Kd = 0.08

                # Integral term
                integral += error

                # Derivative term
                derivative = (
                    error - previous_error
                )

                # PID output
                steering = (
                    Kp * error +
                    Ki * integral +
                    Kd * derivative
                )

                # Save error
                previous_error = error

                # Limit steering
                steering = max(
                    min(steering, 0.3),
                    -0.3
                )

                print(
                    f"Steering: {steering:.2f}"
                )

            # =========================
            # VEHICLE CONTROL
            # =========================

            control = carla.VehicleControl()

            # Lower speed
            control.throttle = 0.15

            # Apply steering
            control.steer = float(steering)

            vehicle.apply_control(control)

            # =========================
            # DISPLAY
            # =========================

            cv2.imshow(
                "PID Autonomous Driving",
                combo
            )

            cv2.imshow(
                "Edges",
                cropped_edges
            )

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    pass

finally:

    print("Cleaning Up")

    camera.stop()

    camera.destroy()

    vehicle.destroy()

    cv2.destroyAllWindows()

    print("Done")