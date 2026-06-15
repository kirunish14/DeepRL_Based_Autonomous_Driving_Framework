import carla
import random
import time

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get world
world = client.get_world()

# Blueprint library
blueprint_library = world.get_blueprint_library()

# Select Tesla Model 3
vehicle_bp = blueprint_library.filter('model3')[0]

# Get spawn points
spawn_points = world.get_map().get_spawn_points()

# Choose random spawn point
spawn_point = random.choice(spawn_points)

# Spawn vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

print("Tesla Model 3 Spawned!")

# Enable autopilot
vehicle.set_autopilot(True)

print("Autopilot Enabled!")

# Keep script alive
try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Destroying vehicle")
    vehicle.destroy()
    print("Done")
