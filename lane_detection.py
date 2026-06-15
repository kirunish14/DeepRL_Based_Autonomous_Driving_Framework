import carla
import random
import cv2
import numpy as np
import time

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

# Slower speed for testing
vehicle.set_autopilot(True)

print("Vehicle Spawned")

# Camera setup
camera_bp = blueprint_library.find('sensor.camera.rgb')

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

# Process image
def process_image(image):
    global current_image

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3]

camera.listen(process_image)

# Region of Interest
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

    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

try:

    while True:

        if current_image is not None:

            frame = current_image.copy()

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blur, 50, 150)

            # Region masking
            cropped_edges = region_of_interest(edges)

            # Detect lines
            lines = cv2.HoughLinesP(
                cropped_edges,
                2,
                np.pi / 180,
                100,
                np.array([]),
                minLineLength=40,
                maxLineGap=5
            )

            # Draw lane lines
            line_image = np.zeros_like(frame)

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

            # Overlay lines
            combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            # Show windows
            cv2.imshow("Lane Detection", combo)
            cv2.imshow("Edges", cropped_edges)

        # Quit with Q
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
