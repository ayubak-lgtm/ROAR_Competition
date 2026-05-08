from functools import reduce
import json
import os
import math
import atexit
from typing import List

import numpy as np
import roar_py_interface

from LateralController import LatController
from ThrottleController import ThrottleController


DEBUG_ENABLED = True
DEBUG_PRINTING = False
debugData = {}


def waypoint_distance(location, waypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def update_waypoint_index(location, current_idx, waypoints):
    for i in range(current_idx, current_idx + len(waypoints)):
        idx = i % len(waypoints)
        if waypoint_distance(location, waypoints[idx]) < 3:
            return idx
    return current_idx


def closest_waypoint_index(location, waypoints):
    best_dist = 100
    best_idx = 0

    for i, waypoint in enumerate(waypoints):
        dist = waypoint_distance(location, waypoint)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx % len(waypoints)


@atexit.register
def save_debug_data():
    if not DEBUG_ENABLED:
        return

    print("Saving debug data")

    debug_folder = os.path.join(os.path.dirname(__file__), "debugData")
    os.makedirs(debug_folder, exist_ok=True)

    debug_path = os.path.join(debug_folder, "debugData.json")

    with open(debug_path, "w+") as outfile:
        outfile.write(json.dumps(debugData, indent=4))

    print("Debug Data Saved")


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:

        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle

        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()

        self.section_indices = []
        self.current_section = 0

        self.current_waypoint_idx = 0
        self.num_ticks = 0
        self.section_start_ticks = 0
        self.lap_num = 1

    async def initialize(self) -> None:
        waypoint_path = os.path.join(
            os.path.dirname(__file__),
            "waypoints",
            "ayub_monza_spicy_line.npz",
        )

        self.maneuverable_waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
            np.load(waypoint_path)
        )

        section_locations = [
            [-278, 372],
            [64, 890],
            [511, 1037],
            [762, 908],
            [198, 307],
            [-11, 60],
            [-85, -339],
            [-210, -1060],
            [-318, -991],
            [-352, -119],
        ]

        self.section_indices = [
            closest_waypoint_index(location, self.maneuverable_waypoints)
            for location in section_locations
        ]

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indices}")
        print("\nLap 1\n")

        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_idx = update_waypoint_index(
            vehicle_location,
            0,
            self.maneuverable_waypoints,
        )

    async def step(self) -> None:
        self.num_ticks += 1

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        current_speed_kmh = np.linalg.norm(vehicle_velocity) * 3.6

        self.current_waypoint_idx = update_waypoint_index(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints,
        )

        self.update_section()

        next_waypoint_index = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.get_smoothed_target(current_speed_kmh)

        steer_control = self.lat_controller.run(
            vehicle_location,
            vehicle_rotation,
            waypoint_to_follow,
        )

        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            next_waypoint_index : next_waypoint_index + 300
        ]

        throttle, brake, gear = self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
        )

        steer_multiplier = self.get_steer_multiplier(current_speed_kmh)

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": np.clip(steer_control * steer_multiplier, -1, 1),
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,
        }

        self.record_debug(
            vehicle_location,
            control,
            current_speed_kmh,
            waypoint_to_follow,
            next_waypoint_index,
        )

        await self.vehicle.apply_action(control)
        return control

    def update_section(self):
        for i, section_idx in enumerate(self.section_indices):
            if abs(self.current_waypoint_idx - section_idx) <= 2 and i != self.current_section:
                print(f"Section {i}: {self.num_ticks - self.section_start_ticks} ticks")

                self.section_start_ticks = self.num_ticks
                self.current_section = i

                if self.current_section == 0 and self.lap_num != 3:
                    self.lap_num += 1
                    print(f"\nLap {self.lap_num}\n")

    def get_lookahead_value(self, speed):
        speed_to_lookahead = {
            90: 9,
            110: 11,
            130: 14,
            160: 18,
            180: 22,
            200: 26,
            250: 30,
            300: 35,
        }

        for speed_limit, lookahead in speed_to_lookahead.items():
            if speed < speed_limit:
                return lookahead

        return 8

    def get_lookahead_index(self, speed):
        return (
            self.current_waypoint_idx + self.get_lookahead_value(speed)
        ) % len(self.maneuverable_waypoints)

    def get_smoothed_target(self, current_speed):
        if 70 < current_speed < 300:
            return self.average_future_waypoints(current_speed)

        return self.maneuverable_waypoints[
            self.get_lookahead_index(current_speed)
        ]

    def average_future_waypoints(self, current_speed):
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)

        num_points = lookahead_value * 2

        if self.current_section == 0:
            num_points = round(lookahead_value * 1.7)

        if self.current_section == 3:
            next_waypoint_index = self.current_waypoint_idx + 22
            num_points = 35

        if self.current_section == 4:
            next_waypoint_index = self.current_waypoint_idx + 24
            num_points = lookahead_value + 5

        if self.current_section == 5:
            num_points = lookahead_value

        if self.current_section == 6:
            next_waypoint_index = self.current_waypoint_idx + 28
            num_points = 5

        if self.current_section == 7:
            num_points = round(lookahead_value * 1.25)

        if self.current_section == 9:
            num_points = 0

        next_waypoint_index %= len(self.maneuverable_waypoints)

        if num_points <= 3:
            return self.maneuverable_waypoints[next_waypoint_index]

        start_index = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        sample_indices = [
            (start_index + i) % len(self.maneuverable_waypoints)
            for i in range(num_points)
        ]

        location_sum = reduce(
            lambda x, y: x + y,
            (self.maneuverable_waypoints[i].location for i in sample_indices),
        )

        averaged_location = location_sum / len(sample_indices)
        original_location = self.maneuverable_waypoints[next_waypoint_index].location

        shift = averaged_location - original_location
        shift_distance = np.linalg.norm(shift)

        max_shift_distance = 2.0

        if self.current_section == 1:
            max_shift_distance = 0.2

        if shift_distance > max_shift_distance:
            direction = shift / shift_distance
            averaged_location = original_location + direction * max_shift_distance

        return roar_py_interface.RoarPyWaypoint(
            location=averaged_location,
            roll_pitch_yaw=np.array([0.0, 0.0, 0.0]),
            lane_width=0.0,
        )

    def get_steer_multiplier(self, current_speed_kmh):
        multiplier = round((current_speed_kmh + 0.001) / 120, 3)

        if self.current_section == 2:
            multiplier *= 1.2

        if self.current_section == 3:
            multiplier = np.clip(multiplier * 1.75, 2.3, 3.5)

        if self.current_section == 4:
            multiplier = min(1.45, multiplier * 1.65)

        if self.current_section == 5:
            multiplier *= 1.1

        if self.current_section == 6:
            multiplier = np.clip(multiplier * 5.5, 5.5, 7)

        if self.current_section == 7:
            multiplier *= 2

        if self.current_section == 9:
            multiplier = max(multiplier, 1.6)

        return multiplier

    def record_debug(
        self,
        vehicle_location,
        control,
        current_speed_kmh,
        waypoint_to_follow,
        target_idx,
    ):
        if not DEBUG_ENABLED:
            return

        debugData[self.num_ticks] = {
            "loc": [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ],
            "throttle": round(float(control["throttle"]), 3),
            "brake": round(float(control["brake"]), 3),
            "steer": round(float(control["steer"]), 10),
            "speed": round(current_speed_kmh, 3),
            "lap": self.lap_num,
        }

        if DEBUG_PRINTING and self.num_ticks % 5 == 0:
            distance_to_target = math.sqrt(
                (waypoint_to_follow.location[0] - vehicle_location[0]) ** 2
                + (waypoint_to_follow.location[1] - vehicle_location[1]) ** 2
            )

            print(
                f"Target waypoint: ({waypoint_to_follow.location[0]:.2f}, "
                f"{waypoint_to_follow.location[1]:.2f}) index {target_idx}\n"
                f"Current location: ({vehicle_location[0]:.2f}, "
                f"{vehicle_location[1]:.2f}) index {self.current_waypoint_idx} "
                f"section {self.current_section}\n"
                f"Distance to target waypoint: {distance_to_target:.3f}\n"
                f"Speed: {current_speed_kmh:.2f} kph\n"
                f"Throttle: {control['throttle']:.3f}\n"
                f"Brake: {control['brake']:.3f}\n"
                f"Steer: {control['steer']:.10f}\n"
            )