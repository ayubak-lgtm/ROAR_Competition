import math
import numpy as np
from scipy.optimize import minimize


def normalize_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MPCLateralController:
    def __init__(self):
        self.wheelbase = 4.7

        self.dt = 0.08
        self.N = 10

        self.max_steer = 1.0
        self.max_steer_angle = 0.90

        self.position_weight = 1.0
        self.heading_weight = 5.0
        self.steer_weight = 0.15
        self.steer_rate_weight = 4.0

        self.prev_solution = None

    def vehicle_model(self, state, steer_cmd, speed_mps):
        x, y, yaw = state

        steer_angle = steer_cmd * self.max_steer_angle

        x_next = x + speed_mps * math.cos(yaw) * self.dt
        y_next = y + speed_mps * math.sin(yaw) * self.dt
        yaw_next = yaw + (speed_mps / self.wheelbase) * math.tan(steer_angle) * self.dt

        return np.array([
            x_next,
            y_next,
            normalize_rad(yaw_next)
        ])

    def build_reference_path(self, waypoints, current_idx, vehicle_speed_kmh):
        refs = []

        if vehicle_speed_kmh < 120:
            step = 1
        elif vehicle_speed_kmh < 150:
            step = 2
        elif vehicle_speed_kmh < 180:
            step = 3
        else:
            step = 5

        for i in range(self.N):
            idx = (current_idx + (i + 1) * step) % len(waypoints)

            p = waypoints[idx].location[:2]

            next_idx = (idx + step) % len(waypoints)
            p_next = waypoints[next_idx].location[:2]

            yaw_ref = math.atan2(
                p_next[1] - p[1],
                p_next[0] - p[0]
            )

            refs.append([p[0], p[1], yaw_ref])

        return np.array(refs)

    def cost_function(self, steer_sequence, initial_state, refs, speed_mps):
        state = np.array(initial_state)
        cost = 0.0

        previous_steer = steer_sequence[0]

        for i in range(self.N):
            steer = steer_sequence[i]
            state = self.vehicle_model(state, steer, speed_mps)

            x, y, yaw = state
            x_ref, y_ref, yaw_ref = refs[i]

            position_error = (x - x_ref) ** 2 + (y - y_ref) ** 2
            heading_error = normalize_rad(yaw - yaw_ref) ** 2
            steer_effort = steer ** 2
            steer_rate = (steer - previous_steer) ** 2

            cost += self.position_weight * position_error
            cost += self.heading_weight * heading_error
            cost += self.steer_weight * steer_effort
            cost += self.steer_rate_weight * steer_rate

            previous_steer = steer

        return cost

    def run(
        self,
        vehicle_location,
        vehicle_rotation,
        vehicle_speed_kmh,
        waypoints,
        current_waypoint_idx
    ):
        x = vehicle_location[0]
        y = vehicle_location[1]
        yaw = vehicle_rotation[2]

        speed_mps = max(vehicle_speed_kmh / 3.6, 1.0)

        initial_state = np.array([x, y, yaw])

        refs = self.build_reference_path(
            waypoints,
            current_waypoint_idx,
            vehicle_speed_kmh
        )

        if self.prev_solution is None:
            initial_guess = np.zeros(self.N)
        else:
            initial_guess = np.roll(self.prev_solution, -1)
            initial_guess[-1] = self.prev_solution[-1]

        bounds = [(-self.max_steer, self.max_steer)] * self.N

        result = minimize(
            self.cost_function,
            initial_guess,
            args=(initial_state, refs, speed_mps),
            bounds=bounds,
            method="SLSQP",
            options={
                "maxiter": 25,
                "ftol": 1e-2,
                "disp": False,
            }
        )

        if result.success:
            self.prev_solution = result.x
            steer = result.x[0]
        else:
            steer = 0.0

        return float(np.clip(-steer, -1.0, 1.0))