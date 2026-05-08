import math
import numpy as np


def wrap_angle(angle: float):
    return angle % (2 * np.pi)


class LatController:
    def __init__(self):
        self.wheelbase = 4.7
        self.steer_gain = 1.5

    def run(
        self,
        vehicle_location,
        vehicle_rotation,
        target_waypoint,
    ) -> float:

        vehicle_xy = np.array(vehicle_location[:2])
        target_xy = np.array(target_waypoint.location[:2])

        direction_vector = target_xy - vehicle_xy

        lookahead_distance = np.linalg.norm(direction_vector)

        if lookahead_distance < 1e-6:
            return 0.0

        direction_unit_vector = (
            direction_vector / lookahead_distance
        )

        target_heading = math.atan2(
            direction_unit_vector[1],
            direction_unit_vector[0]
        )

        vehicle_heading = wrap_angle(vehicle_rotation[2])

        heading_error = (
            vehicle_heading - wrap_angle(target_heading)
        )

        steering_angle = math.atan2(
            2.0
            * self.wheelbase
            * math.sin(heading_error),
            lookahead_distance
        )

        steering_command = (
            self.steer_gain * steering_angle
        )

        return float(steering_command)