import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
import os


async def main():
    carla_client = carla.Client("localhost", 2000)
    carla_client.set_timeout(10.0)

    roar_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_instance.world

    world.set_asynchronous(True)
    world.set_control_steps(0.0, 0.005)

    print("Map Name:", world.map_name)

    waypoints = world.maneuverable_waypoints

    output_folder = os.path.join(os.path.dirname(__file__), "waypoints")
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(output_folder, "Monza Original Waypoints.npz")

    np.savez_compressed(
        file_path,
        **roar_py_interface.RoarPyWaypoint.save_waypoint_list(waypoints),
    )

    print("\nSaved base waypoints to:")
    print(file_path)

    roar_instance.close()


if __name__ == "__main__":
    asyncio.run(main())