import os
import asyncio
from typing import List, Optional, Dict, Any

import carla
import pygame
import numpy as np
from PIL.Image import Image

import roar_py_carla
import roar_py_interface


WAYPOINT_DISTANCE = 2.0
WAYPOINT_LANE_WIDTH = 12.0
OUTPUT_FOLDER = "waypoints"
OUTPUT_FILE = "ayub_monza_spicy_line.npz"


class ManualControlViewer:
    def __init__(self):
        self.screen = None
        self.clock = None

    def init_pygame(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Ayub Waypoint Drip Collector")
        self.clock = pygame.time.Clock()

    def render(self, image: roar_py_interface.RoarPyCameraSensorData):
        image_pil: Image = image.get_image()

        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)

        control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None

        keys = pygame.key.get_pressed()

        # UP = throttle
        if keys[pygame.K_UP]:
            control["throttle"] = 0.6

        # DOWN = brake
        if keys[pygame.K_DOWN]:
            control["brake"] = 0.6

        # LEFT = steer left
        if keys[pygame.K_LEFT]:
            control["steer"] = -0.25

        # RIGHT = steer right
        if keys[pygame.K_RIGHT]:
            control["steer"] = 0.25

        # A/D = fine steering
        if keys[pygame.K_a]:
            control["steer"] = -0.08

        if keys[pygame.K_d]:
            control["steer"] = 0.08

        # W = slow throttle (moved from S)
        if keys[pygame.K_w]:
            control["throttle"] = 0.25

        # S = reverse (moved from R)
        if keys[pygame.K_s]:
            control["reverse"] = 1
            control["throttle"] = 0.4

        image_surface = pygame.image.fromstring(
            image_pil.tobytes(), image_pil.size, image_pil.mode
        ).convert()

        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)

        return control, keys


async def main():
    carla_client = carla.Client("localhost", 2000)
    carla_client.set_timeout(10.0)

    roar_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_instance.world

    world.set_asynchronous(True)
    world.set_control_steps(0.0, 0.005)

    print("\nAvailable spawn points:")
    for i, sp in enumerate(world.spawn_points):
        print(i, sp[0])

    spawn_num = int(input("\nChoose spawn point number: "))
    spawn_point, spawn_rpy = world.spawn_points[spawn_num]

    print("\nSpawning car at:", spawn_point)

    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point,
        spawn_rpy,
    )

    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([
            -2.0 * vehicle.bounding_box.extent[0],
            0.0,
            3.0 * vehicle.bounding_box.extent[2],
        ]),
        np.array([0.0, 10 / 180.0 * np.pi, 0.0]),
    )

    viewer = ManualControlViewer()

    waypoints: List[roar_py_interface.RoarPyWaypoint] = []

    first_waypoint = roar_py_interface.RoarPyWaypoint(
        spawn_point,
        spawn_rpy,
        WAYPOINT_LANE_WIDTH,
    )

    waypoints.append(first_waypoint)

    collecting = True

    print("\nControls:")
    print("UP/DOWN/LEFT/RIGHT = drive")
    print("A/D = fine steering")
    print("W = slow throttle")
    print("S = reverse")
    print("SPACE = pause/resume waypoint collection")
    print("ESC = stop and save\n")

    try:
        while True:
            await world.step()

            current_location = vehicle.get_3d_location()
            current_rpy = vehicle.get_roll_pitch_yaw()

            last_waypoint = waypoints[-1]
            distance_to_last = np.linalg.norm(
                current_location[:2] - last_waypoint.location[:2]
            )

            if collecting and distance_to_last > WAYPOINT_DISTANCE:
                new_waypoint = roar_py_interface.RoarPyWaypoint(
                    current_location - vehicle.bounding_box.extent[2] * np.array([0, 0, 1]),
                    current_rpy,
                    WAYPOINT_LANE_WIDTH,
                )

                waypoints.append(new_waypoint)

                if len(waypoints) % 50 == 0:
                    print(f"Collected {len(waypoints)} waypoints")

            image = await camera.receive_observation()
            control, keys = viewer.render(image)

            if control is None:
                break

            # SPACE → toggle recording
            if keys[pygame.K_SPACE]:
                collecting = not collecting
                print("Collecting:", collecting)
                await asyncio.sleep(0.3)

            # ESC → stop and save
            if keys[pygame.K_ESCAPE]:
                print("\nStopping and saving waypoints...")
                break

            await vehicle.apply_action(control)

    finally:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

        np.savez_compressed(
            output_path,
            **roar_py_interface.RoarPyWaypoint.save_waypoint_list(waypoints),
        )

        print(f"\nSaved {len(waypoints)} waypoints to:")
        print(output_path)

        roar_instance.close()
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())