import carla
import random
import cv2
import numpy as np
import time

# =========================
# GLOBAL VARIABLES
# =========================

current_image = None

# Steering smoothing
previous_steering = 0.0

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

camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
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

camera.listen(process_image)

# =========================
# ROI
# =========================

def region_of_interest(img):

    height = img.shape[0]

    polygons = np.array([
        [
            (0, height),
            (640, height),
            (420, 250),
            (220, 250)
        ]
    ])

    mask = np.zeros_like(img)

    cv2.fillPoly(mask, polygons, 255)

    masked = cv2.bitwise_and(img, mask)

    return masked

# =========================
# AVERAGE SLOPE
# =========================

def average_slope_intercept(frame, lines):

    left_fit = []
    right_fit = []

    if lines is None:
        return None, None

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        parameters = np.polyfit(
            (x1, x2),
            (y1, y2),
            1
        )

        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))

        else:
            right_fit.append((slope, intercept))

    left_lane = None
    right_lane = None

    if left_fit:
        left_lane = np.average(
            left_fit,
            axis=0
        )

    if right_fit:
        right_lane = np.average(
            right_fit,
            axis=0
        )

    return left_lane, right_lane

# =========================
# CREATE LINE POINTS
# =========================

def make_points(frame, line):

    if line is None:
        return None

    slope, intercept = line

    # Avoid division problems
    if abs(slope) < 0.1:
        return None

    y1 = frame.shape[0]
    y2 = int(y1 * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # Prevent invalid huge coordinates
    x1 = max(min(x1, 2000), -2000)
    x2 = max(min(x2, 2000), -2000)

    return np.array([x1, y1, x2, y2], dtype=np.int32)

# =========================
# MAIN LOOP
# =========================

try:

    while True:

        if current_image is not None:

            frame = current_image.copy()

            # Grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Blur
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

            # ROI
            cropped_edges = region_of_interest(
                edges
            )

            # Hough lines
            lines = cv2.HoughLinesP(
                cropped_edges,
                2,
                np.pi / 180,
                100,
                np.array([]),
                minLineLength=40,
                maxLineGap=20
            )

            # Average lanes
            left_lane, right_lane = average_slope_intercept(
                frame,
                lines
            )

            left_line = make_points(
                frame,
                left_lane
            )

            right_line = make_points(
                frame,
                right_lane
            )

            line_image = np.zeros_like(frame)

            lane_center = None

            # Draw left lane
            if left_line is not None:

                x1, y1, x2, y2 = left_line

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    10
                )

            # Draw right lane
            if right_line is not None:

                x1, y1, x2, y2 = right_line

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    10
                )

            combo = cv2.addWeighted(
                frame,
                0.8,
                line_image,
                1,
                1
            )

            # =========================
            # STEERING LOGIC
            # =========================

            steering = previous_steering

            if left_line is not None and right_line is not None:

                left_x2 = left_line[2]
                right_x2 = right_line[2]

                lane_center = (
                    left_x2 + right_x2
                ) // 2

                frame_center = 320

                deviation = lane_center - frame_center

                steering = deviation / 320

                # Smooth steering
                steering = (
                    previous_steering * 0.7 +
                    steering * 0.3
                )

                # Clamp steering
                steering = max(
                    min(steering, 0.25),
                    -0.25
                )

                previous_steering = steering

                # Draw centers
                cv2.circle(
                    combo,
                    (lane_center, 300),
                    10,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    combo,
                    (frame_center, 300),
                    10,
                    (255, 0, 0),
                    -1
                )

                print(
                    f"Steering: {steering:.2f}"
                )

            # =========================
            # VEHICLE CONTROL
            # =========================

            control = carla.VehicleControl()

            control.throttle = 0.18

            control.steer = float(steering)

            vehicle.apply_control(control)

            # =========================
            # DISPLAY
            # =========================

            cv2.imshow(
                "Advanced Autonomous Driving",
                combo
            )

            cv2.imshow(
                "Edges",
                cropped_edges
            )

        # Quit
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