"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List
import roar_py_interface
import numpy as np
import matplotlib.pyplot as plt
import time


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint]
) -> int:
    def dist_to_waypoint(waypoint: roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(location[:2] - waypoint.location[:2])

    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


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
        self.lap_count = 0
        self.prev_wp_idx = 10
        self.lap_start_time = time.time()

        # longitudinal PID memory
        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0

        # lateral PID memory
        self.heading_error_integral = 0.0
        self.prev_heading_error = 0.0

        # known crash region from logs
        self.crash_wp_idx = 1375

    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # ---------------------------------------------------
        # Extract centerline
        # ---------------------------------------------------
        waypoint_xy = np.array([
            wp.location[:2] for wp in self.maneuverable_waypoints
        ])
        self.centerline = waypoint_xy

        x = waypoint_xy[:, 0]
        y = waypoint_xy[:, 1]

        # ---------------------------------------------------
        # Plot centerline
        # ---------------------------------------------------
        plt.figure(figsize=(10, 10))
        plt.plot(x, y, 'b-', linewidth=1.5, label='Centerline Waypoints')
        plt.scatter(x[0], y[0], c='green', s=80, label='Start')
        plt.scatter(
            x[self.current_waypoint_idx],
            y[self.current_waypoint_idx],
            c='red',
            s=80,
            label='Current Index'
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Monza Track Centerline from Maneuverable Waypoints")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("track_centerline.png", dpi=200)
        plt.close()

        # ---------------------------------------------------
        # Approximate boundaries assuming total track width = 5 m
        # ---------------------------------------------------
        track_width = 5.0
        half_width = track_width / 2.0

        n = len(waypoint_xy)
        left_boundary = []
        right_boundary = []

        for i in range(n):
            p_prev = waypoint_xy[(i - 1) % n]
            p_next = waypoint_xy[(i + 1) % n]

            tangent = p_next - p_prev
            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm < 1e-6:
                tangent_unit = np.array([1.0, 0.0])
            else:
                tangent_unit = tangent / tangent_norm

            normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])

            p_center = waypoint_xy[i]
            p_left = p_center + half_width * normal_unit
            p_right = p_center - half_width * normal_unit

            left_boundary.append(p_left)
            right_boundary.append(p_right)

        self.left_boundary = np.array(left_boundary)
        self.right_boundary = np.array(right_boundary)

        # ---------------------------------------------------
        # Plot full boundaries
        # ---------------------------------------------------
        plt.figure(figsize=(10, 10))
        plt.plot(
            waypoint_xy[:, 0],
            waypoint_xy[:, 1],
            color='blue',
            linewidth=1.0,
            alpha=0.5,
            label='Centerline'
        )
        plt.plot(
            self.left_boundary[:, 0],
            self.left_boundary[:, 1],
            color='red',
            linewidth=2.0,
            linestyle='--',
            label='Left Boundary'
        )
        plt.plot(
            self.right_boundary[:, 0],
            self.right_boundary[:, 1],
            color='green',
            linewidth=2.0,
            linestyle='--',
            label='Right Boundary'
        )
        plt.scatter(
            waypoint_xy[0, 0],
            waypoint_xy[0, 1],
            c='black',
            s=60,
            label='Start'
        )
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Approximate Track Boundaries (Width = 5 m)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("track_boundaries.png", dpi=300)
        plt.close()

        # ---------------------------------------------------
        # Plot zoomed boundaries near start/current region
        # ---------------------------------------------------
        x0 = waypoint_xy[self.current_waypoint_idx, 0]
        y0 = waypoint_xy[self.current_waypoint_idx, 1]

        plt.figure(figsize=(8, 8))
        plt.plot(
            waypoint_xy[:, 0],
            waypoint_xy[:, 1],
            color='blue',
            linewidth=1.0,
            alpha=0.5,
            label='Centerline'
        )
        plt.plot(
            self.left_boundary[:, 0],
            self.left_boundary[:, 1],
            color='red',
            linewidth=2.5,
            linestyle='--',
            label='Left Boundary'
        )
        plt.plot(
            self.right_boundary[:, 0],
            self.right_boundary[:, 1],
            color='green',
            linewidth=2.5,
            linestyle='--',
            label='Right Boundary'
        )
        plt.scatter(
            waypoint_xy[self.current_waypoint_idx, 0],
            waypoint_xy[self.current_waypoint_idx, 1],
            c='black',
            s=80,
            label='Current Region'
        )
        plt.xlim(x0 - 40, x0 + 40)
        plt.ylim(y0 - 40, y0 + 40)
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Zoomed Track Boundaries Near Start Region")
        plt.legend()
        plt.tight_layout()
        plt.savefig("track_boundaries_zoom.png", dpi=300)
        plt.close()

        # ---------------------------------------------------
        # Compute curvature profile
        # ---------------------------------------------------
        curvatures = []
        gap = 5

        for i in range(n):
            p1 = waypoint_xy[i % n]
            p2 = waypoint_xy[(i + gap) % n]
            p3 = waypoint_xy[(i + 2 * gap) % n]

            v1 = p2 - p1
            v2 = p3 - p2

            h1 = np.arctan2(v1[1], v1[0])
            h2 = np.arctan2(v2[1], v2[0])

            curvature = abs(normalize_rad(h2 - h1))
            curvatures.append(curvature)

        self.curvatures = np.array(curvatures)

        # ---------------------------------------------------
        # Plot curvature-colored track
        # ---------------------------------------------------
        plt.figure(figsize=(10, 10))
        sc = plt.scatter(
            waypoint_xy[:, 0],
            waypoint_xy[:, 1],
            c=self.curvatures,
            cmap='jet',
            s=8
        )
        plt.colorbar(sc, label="Curvature")
        plt.scatter(
            waypoint_xy[0, 0],
            waypoint_xy[0, 1],
            c='black',
            s=60,
            label='Start'
        )
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Track Colored by Curvature")
        plt.legend()
        plt.tight_layout()
        plt.savefig("track_curvature.png", dpi=300)
        plt.close()

        # ---------------------------------------------------
        # Target speed profile from curvature
        # ---------------------------------------------------
        k = 8.0
        epsilon = 1e-3
        target_speeds = k / np.sqrt(self.curvatures + epsilon)
        target_speeds = np.clip(target_speeds, 8.0, 40.0)

        window = 21
        pad = window // 2
        padded = np.pad(target_speeds, (pad, pad), mode='wrap')
        self.target_speeds = np.convolve(
            padded,
            np.ones(window) / window,
            mode='valid'
        )

        # ---------------------------------------------------
        # Plot target speed profile
        # ---------------------------------------------------
        plt.figure(figsize=(10, 10))
        sc = plt.scatter(
            waypoint_xy[:, 0],
            waypoint_xy[:, 1],
            c=self.target_speeds,
            cmap='jet',
            s=8
        )
        plt.colorbar(sc, label="Target Speed (m/s)")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Target Speed Profile from Curvature")
        plt.tight_layout()
        plt.savefig("track_speed_profile.png", dpi=300)
        plt.close()

        # ---------------------------------------------------
        # Crash region plots
        # ---------------------------------------------------
        plt.figure(figsize=(10, 10))
        plt.plot(waypoint_xy[:, 0], waypoint_xy[:, 1], 'b-', linewidth=1.5, label='Centerline')
        plt.scatter(
            waypoint_xy[self.crash_wp_idx, 0],
            waypoint_xy[self.crash_wp_idx, 1],
            c='magenta',
            s=120,
            label=f'Crash Region WP {self.crash_wp_idx}'
        )
        plt.scatter(
            waypoint_xy[0, 0],
            waypoint_xy[0, 1],
            c='green',
            s=80,
            label='Start'
        )
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Crash Region on Track")
        plt.legend()
        plt.tight_layout()
        plt.savefig("track_crash_region.png", dpi=300)
        plt.close()

        x_c = waypoint_xy[self.crash_wp_idx, 0]
        y_c = waypoint_xy[self.crash_wp_idx, 1]

        plt.figure(figsize=(8, 8))
        plt.plot(waypoint_xy[:, 0], waypoint_xy[:, 1], 'b-', linewidth=1.0, alpha=0.5, label='Centerline')
        plt.plot(self.left_boundary[:, 0], self.left_boundary[:, 1], 'r--', linewidth=2.0, label='Left Boundary')
        plt.plot(self.right_boundary[:, 0], self.right_boundary[:, 1], 'g--', linewidth=2.0, label='Right Boundary')
        plt.scatter(
            waypoint_xy[self.crash_wp_idx, 0],
            waypoint_xy[self.crash_wp_idx, 1],
            c='magenta',
            s=120,
            label=f'Crash WP {self.crash_wp_idx}'
        )
        plt.xlim(x_c - 40, x_c + 40)
        plt.ylim(y_c - 40, y_c + 40)
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Zoomed Crash Region")
        plt.legend()
        plt.tight_layout()
        plt.savefig("track_crash_region_zoom.png", dpi=300)
        plt.close()

        print("Initialization complete and crash-region plots saved")

    async def step(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_speed = np.linalg.norm(vehicle_velocity)

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        current_curvature = self.curvatures[self.current_waypoint_idx]

        # Shorter lookahead in tighter corners
        if current_curvature > 0.12:
            lookahead = 2
        elif current_curvature > 0.08:
            lookahead = 3
        elif current_curvature > 0.04:
            lookahead = 4
        else:
            if vehicle_speed < 10:
                lookahead = 4
            elif vehicle_speed < 20:
                lookahead = 6
            elif vehicle_speed < 30:
                lookahead = 8
            else:
                lookahead = 9

        waypoint_to_follow = self.maneuverable_waypoints[
            (self.current_waypoint_idx + lookahead) % len(self.maneuverable_waypoints)
        ]

        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(
            vector_to_waypoint[1],
            vector_to_waypoint[0]
        )

        delta_heading = normalize_rad(
            heading_to_waypoint - vehicle_rotation[2]
        )

        # Stronger steering in tight corners
        if current_curvature > 0.08:
            kp_steer = 3
            ki_steer = 0.001
            kd_steer = 2
        else:
            kp_steer = 1.6
            ki_steer = 0.002
            kd_steer = 0.8

        speed_scale = 1.0 / max(np.sqrt(vehicle_speed), 1.0)

        self.heading_error_integral += delta_heading
        self.heading_error_integral = np.clip(self.heading_error_integral, -1.0, 1.0)

        heading_error_derivative = delta_heading - self.prev_heading_error
        self.prev_heading_error = delta_heading

        steer_control = -(
            kp_steer * delta_heading
            + ki_steer * self.heading_error_integral
            + kd_steer * heading_error_derivative
        ) * speed_scale

        steer_control = np.clip(steer_control, -1.0, 1.0)

        target_speed = self.target_speeds[self.current_waypoint_idx]

        if current_curvature > 0.10:
            target_speed *= 0.80
        elif current_curvature > 0.05:
            target_speed *= 0.90

        speed_error = target_speed - vehicle_speed

        kp_speed = 0.18
        ki_speed = 0.002
        kd_speed = 0.08

        self.speed_error_integral += speed_error
        self.speed_error_integral = np.clip(self.speed_error_integral, -50.0, 50.0)

        speed_error_derivative = speed_error - self.prev_speed_error
        self.prev_speed_error = speed_error

        speed_control = (
            kp_speed * speed_error
            + ki_speed * self.speed_error_integral
            + kd_speed * speed_error_derivative
        )

        if speed_control >= 0:
            throttle_control = np.clip(speed_control, 0.0, 1.0)
            brake_control = 0.0
        else:
            throttle_control = 0.0
            brake_control = np.clip(-speed_control, 0.0, 1.0)

        if current_curvature > 0.08:
            print(
                f"wp={self.current_waypoint_idx}, "
                f"curv={current_curvature:.3f}, "
                f"lookahead={lookahead}, "
                f"speed={vehicle_speed:.2f}, "
                f"target_speed={target_speed:.2f}, "
                f"delta_heading={delta_heading:.3f}, "
                f"steer={steer_control:.3f}"
            )

        control = {
            "throttle": throttle_control,
            "steer": steer_control,
            "brake": brake_control,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        # Detect lap completion
        if self.current_waypoint_idx < self.prev_wp_idx:
            current_time = time.time()
            lap_time = current_time - self.lap_start_time

            self.lap_count += 1
            print(f"\n🏁 LAP {self.lap_count} COMPLETED | Time = {lap_time:.2f} sec\n")

            self.lap_start_time = current_time

        self.prev_wp_idx = self.current_waypoint_idx

        await self.vehicle.apply_action(control)
        return control