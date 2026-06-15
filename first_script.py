import sys
import glob

try:
    sys.path.append(glob.glob('/home/kirunish/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()

print("Connected to CARLA")
print(world)
