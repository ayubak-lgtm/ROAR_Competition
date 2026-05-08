import math
from collections import deque

import numpy as np
import roar_py_interface

from SpeedData import SpeedData


def waypoint_gap(p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])


class ThrottleController:
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self):
        self.max_radius = 10000
        self.max_speed = 300

        self.intended_target_distance = [0, 30, 60, 90, 120, 140, 170]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]

        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2

        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0

        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")

    def run(self, waypoints, current_location, current_speed, current_section):
        self.tick_counter += 1

        throttle, brake = self.compute_control(
            current_location,
            current_speed,
            current_section,
            waypoints,
        )

        gear = max(1, int(current_speed / 60))

        if throttle < 0:
            gear = -1

        self.previous_speed = current_speed

        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        return throttle, brake, gear

    def compute_control(self, current_location, current_speed, current_section, waypoints):
        sampled_waypoints = self.sample_forward_waypoints(
            current_location,
            waypoints,
        )

        close_radius = self.curve_radius(
            sampled_waypoints[self.close_index : self.close_index + 3]
        )

        mid_radius = self.curve_radius(
            sampled_waypoints[self.mid_index : self.mid_index + 3]
        )

        far_radius = self.curve_radius(
            sampled_waypoints[self.far_index : self.far_index + 3]
        )

        close_target_speed = self.radius_to_speed(close_radius, current_section)
        mid_target_speed = self.radius_to_speed(mid_radius, current_section)
        far_target_speed = self.radius_to_speed(far_radius, current_section)

        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]

        speed_options = [
            self.speed_for_turn(close_distance, close_target_speed, current_speed),
            self.speed_for_turn(mid_distance, mid_target_speed, current_speed),
            self.speed_for_turn(far_distance, far_target_speed, current_speed),
        ]

        if current_speed > 100:
            if current_section != 9:
                wide_mid_radius = self.curve_radius(
                    [
                        sampled_waypoints[self.mid_index],
                        sampled_waypoints[self.mid_index + 2],
                        sampled_waypoints[self.mid_index + 4],
                    ]
                )

                wide_mid_speed = self.radius_to_speed(
                    wide_mid_radius,
                    current_section,
                )

                speed_options.append(
                    self.speed_for_turn(
                        close_distance,
                        wide_mid_speed,
                        current_speed,
                    )
                )

            wide_close_radius = self.curve_radius(
                [
                    sampled_waypoints[self.close_index],
                    sampled_waypoints[self.close_index + 3],
                    sampled_waypoints[self.close_index + 6],
                ]
            )

            wide_close_speed = self.radius_to_speed(
                wide_close_radius,
                current_section,
            )

            speed_options.append(
                self.speed_for_turn(
                    close_distance,
                    wide_close_speed,
                    current_speed,
                )
            )

        selected_speed_data = self.pick_most_conservative_speed(speed_options)

        self.log_speed_set(
            " -- SPEED: ",
            speed_options[0].recommended_speed_now,
            speed_options[1].recommended_speed_now,
            speed_options[2].recommended_speed_now,
            0 if len(speed_options) < 4 else speed_options[3].recommended_speed_now,
            current_speed,
        )

        throttle, brake = self.speed_data_to_throttle_and_brake(
            selected_speed_data
        )

        self.dprint(
            "--- throt " + str(throttle) + " brake " + str(brake) + "---"
        )

        return throttle, brake

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        percent_of_max = (
            speed_data.current_speed / speed_data.recommended_speed_now
        )

        avg_speed_change_per_tick = 2.4
        percent_change_per_tick = 0.075

        true_percent_change_per_tick = round(
            avg_speed_change_per_tick / (speed_data.current_speed + 0.001),
            5,
        )

        speed_up_threshold = 0.9
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        brake_threshold_multiplier = 1.0

        percent_speed_change = (
            speed_data.current_speed - self.previous_speed
        ) / (self.previous_speed + 0.0001)

        speed_change = round(
            speed_data.current_speed - self.previous_speed,
            3,
        )

        if percent_of_max > 1:
            if percent_of_max > 1 + (
                brake_threshold_multiplier * true_percent_change_per_tick
            ):
                if self.brake_ticks > 0:
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: counter "
                        + str(self.brake_ticks)
                    )
                    return -1, 1

                if self.brake_ticks <= 0 and speed_change < 2.5:
                    self.brake_ticks = round(
                        (
                            speed_data.current_speed
                            - speed_data.recommended_speed_now
                        )
                        / 3
                    )

                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: initiate counter "
                        + str(self.brake_ticks)
                    )

                    return -1, 1

                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " brake: throttle early1: sp_ch="
                    + str(percent_speed_change)
                )

                self.brake_ticks = 0
                return 1, 0

            if speed_change >= 2.5:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " brake: throttle early2: sp_ch="
                    + str(percent_speed_change)
                )

                self.brake_ticks = 0
                return 1, 0

            throttle_to_maintain = self.get_throttle_to_maintain_speed(
                speed_data.current_speed
            )

            if percent_of_max > 1.02 or percent_speed_change > (
                -true_percent_change_per_tick / 2
            ):
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " brake: throttle down: sp_ch="
                    + str(percent_speed_change)
                )

                return throttle_to_maintain * throttle_decrease_multiple, 0

            return throttle_to_maintain, 0

        self.brake_ticks = 0

        if speed_change >= 2.5:
            self.dprint(
                "tb: tick "
                + str(self.tick_counter)
                + " throttle: full speed drop: sp_ch="
                + str(percent_speed_change)
            )

            return 1, 0

        if percent_of_max < speed_up_threshold:
            self.dprint(
                "tb: tick "
                + str(self.tick_counter)
                + " throttle full: p_max="
                + str(percent_of_max)
            )

            return 1, 0

        throttle_to_maintain = self.get_throttle_to_maintain_speed(
            speed_data.current_speed
        )

        if percent_of_max < 0.98 or true_percent_change_per_tick < -0.01:
            self.dprint(
                "tb: tick "
                + str(self.tick_counter)
                + " throttle up: sp_ch="
                + str(percent_speed_change)
            )

            return throttle_to_maintain * throttle_increase_multiple, 0

        self.dprint(
            "tb: tick "
            + str(self.tick_counter)
            + " throttle maintain: sp_ch="
            + str(percent_speed_change)
        )

        return throttle_to_maintain, 0

    def isSpeedDroppingFast(self, percent_change_per_tick: float, current_speed):
        percent_speed_change = (
            current_speed - self.previous_speed
        ) / (self.previous_speed + 0.0001)

        return percent_speed_change < (-percent_change_per_tick / 2)

    def pick_most_conservative_speed(self, speed_data):
        min_speed = 1000
        selected_index = -1

        for i, speed_state in enumerate(speed_data):
            if speed_state.recommended_speed_now < min_speed:
                min_speed = speed_state.recommended_speed_now
                selected_index = i

        if selected_index != -1:
            return speed_data[selected_index]

        return speed_data[0]

    def get_throttle_to_maintain_speed(self, current_speed: float):
        throttle = 0.75 + current_speed / 500
        return throttle

    def speed_for_turn(
        self,
        distance: float,
        target_speed: float,
        current_speed: float,
    ):
        d = (1 / 675) * (target_speed**2) + distance
        max_speed = math.sqrt(825 * d)

        return SpeedData(
            distance,
            current_speed,
            target_speed,
            max_speed,
        )

    def sample_forward_waypoints(self, current_location, waypoints):
        sampled_points = []
        sampled_distances = []

        start = roar_py_interface.RoarPyWaypoint(
            current_location,
            np.array([0.0, 0.0, 0.0]),
            0.0,
        )

        sampled_points.append(start)

        running_distance = 0

        for waypoint in waypoints:
            running_distance += waypoint_gap(start, waypoint)

            if running_distance > self.intended_target_distance[len(sampled_points)]:
                self.target_distance[len(sampled_points)] = running_distance
                sampled_points.append(waypoint)
                sampled_distances.append(running_distance)

            start = waypoint

            if len(sampled_points) >= len(self.target_distance):
                break

        self.dprint("wp dist " + str(sampled_distances))

        return sampled_points

    def curve_radius(self, waypoints):
        p1 = (waypoints[0].location[0], waypoints[0].location[1])
        p2 = (waypoints[1].location[0], waypoints[1].location[1])
        p3 = (waypoints[2].location[0], waypoints[2].location[1])

        side_1 = round(math.dist(p1, p2), 3)
        side_2 = round(math.dist(p2, p3), 3)
        side_3 = round(math.dist(p1, p3), 3)

        small_num = 2

        if side_1 < small_num or side_2 < small_num or side_3 < small_num:
            return self.max_radius

        semi_perimeter = (side_1 + side_2 + side_3) / 2

        area_squared = (
            semi_perimeter
            * (semi_perimeter - side_1)
            * (semi_perimeter - side_2)
            * (semi_perimeter - side_3)
        )

        if area_squared < small_num:
            return self.max_radius

        radius = (
            side_1
            * side_2
            * side_3
        ) / (4 * math.sqrt(area_squared))

        return radius

    def radius_to_speed(self, radius: float, current_section: int):
        mu = 2.75

        if radius >= self.max_radius:
            return self.max_speed

        if current_section == 2:
            mu = 3.35

        if current_section == 3:
            mu = 3.3

        if current_section == 4:
            mu = 2.85

        if current_section == 6:
            mu = 3.3

        if current_section == 9:
            mu = 2.1

        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6

        return max(20, min(target_speed, self.max_speed))

    def log_speed_set(self, text, s1, s2, s3, s4, current_speed):
        self.dprint(
            text
            + " s1= "
            + str(round(s1, 2))
            + " s2= "
            + str(round(s2, 2))
            + " s3= "
            + str(round(s3, 2))
            + " s4= "
            + str(round(s4, 2))
            + " cspeed= "
            + str(round(current_speed, 2))
        )

    def dprint(self, text):
        if self.display_debug:
            print(text)
            self.debug_strings.append(text)