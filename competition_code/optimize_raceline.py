import os
import math
import numpy as np
from scipy.optimize import minimize
import roar_py_interface


# ============================================================
# Ayub Racing Line Optimizer
# Geometric minimum-curvature optimizer
#
# Input:
#   waypoints/Monza Original Waypoints.npz
#
# Output:
#   waypoints/ayub_optimized_raceline.npz
#
# This does NOT need vehicle mass, tire model, yaw inertia, etc.
# It only creates a smoother racing line inside the lane width.
# ============================================================


INPUT_FILE = "Monza Original Waypoints.npz"
OUTPUT_FILE = "ayub_optimized_raceline.npz"

# Keep margin so the car does not run exactly on the edge
TRACK_MARGIN = 1.5

# Use fewer points for faster optimization.
# 1 = use every waypoint
# 2 = use every 2nd waypoint
# 3 = use every 3rd waypoint
DOWNSAMPLE = 3

# Smoothness penalty. Higher = smoother, less aggressive line.
SMOOTHNESS_WEIGHT = 0.4

# Lateral offset change penalty. Higher = less zig-zag.
OFFSET_CHANGE_WEIGHT = 0.2


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def get_base_path():
    return os.path.abspath(os.path.dirname(__file__))


def load_waypoints(filename):
    path = os.path.join(get_base_path(), "waypoints", filename)
    print("Loading:", path)

    return roar_py_interface.RoarPyWaypoint.load_waypoint_list(
        np.load(path)
    )


def save_waypoints(filename, waypoints):
    output_folder = os.path.join(get_base_path(), "waypoints")
    os.makedirs(output_folder, exist_ok=True)

    path = os.path.join(output_folder, filename)

    np.savez_compressed(
        path,
        **roar_py_interface.RoarPyWaypoint.save_waypoint_list(waypoints),
    )

    print("\nSaved optimized racing line to:")
    print(path)


def compute_normals(points):
    """
    Computes left/right normal direction at each centerline point.
    """
    normals = []

    n = len(points)

    for i in range(n):
        prev_p = points[(i - 1) % n]
        next_p = points[(i + 1) % n]

        tangent = next_p[:2] - prev_p[:2]
        norm = np.linalg.norm(tangent)

        if norm < 1e-6:
            normals.append(np.array([0.0, 1.0]))
            continue

        tangent = tangent / norm

        # Rotate tangent by 90 degrees to get normal
        normal = np.array([-tangent[1], tangent[0]])

        normals.append(normal)

    return np.array(normals)


def curvature_cost(offsets, center_points, normals):
    """
    Minimum curvature objective.
    The optimized racing line is:
        race_point = center_point + offset * normal

    We penalize:
    1. second derivative of path = curvature-like sharpness
    2. sudden lateral offset changes = zig-zag
    """
    race_points_2d = center_points[:, :2] + offsets[:, None] * normals

    cost = 0.0
    n = len(race_points_2d)

    for i in range(n):
        p_prev = race_points_2d[(i - 1) % n]
        p = race_points_2d[i]
        p_next = race_points_2d[(i + 1) % n]

        second_diff = p_next - 2 * p + p_prev
        cost += np.dot(second_diff, second_diff)

    # Penalize offset jumps
    for i in range(n):
        offset_jump = offsets[(i + 1) % n] - offsets[i]
        cost += OFFSET_CHANGE_WEIGHT * offset_jump ** 2

    # Extra smoothing
    for i in range(n):
        offset_second_diff = (
            offsets[(i + 1) % n] - 2 * offsets[i] + offsets[(i - 1) % n]
        )
        cost += SMOOTHNESS_WEIGHT * offset_second_diff ** 2

    return cost


def create_optimized_waypoints(original_waypoints, optimized_points):
    optimized_waypoints = []

    n = len(optimized_points)

    for i in range(n):
        p = optimized_points[i]
        p_next = optimized_points[(i + 1) % n]

        dx = p_next[0] - p[0]
        dy = p_next[1] - p[1]

        yaw = math.atan2(dy, dx)

        location = np.array([p[0], p[1], original_waypoints[i].location[2]])

        roll_pitch_yaw = np.array([0.0, 0.0, yaw])

        optimized_waypoints.append(
            roar_py_interface.RoarPyWaypoint(
                location=location,
                roll_pitch_yaw=roll_pitch_yaw,
                lane_width=original_waypoints[i].lane_width,
            )
        )

    return optimized_waypoints


def main():
    original_waypoints_full = load_waypoints(INPUT_FILE)

    original_waypoints = original_waypoints_full[::DOWNSAMPLE]

    center_points = np.array([wp.location for wp in original_waypoints])
    normals = compute_normals(center_points)

    lane_widths = np.array([
        wp.lane_width if wp.lane_width is not None and wp.lane_width > 0 else 12.0
        for wp in original_waypoints
    ])

    half_widths = lane_widths / 2.0

    bounds = []
    for hw in half_widths:
        usable_half_width = max(0.5, hw - TRACK_MARGIN)
        bounds.append((-usable_half_width, usable_half_width))

    initial_offsets = np.zeros(len(original_waypoints))

    print("\nStarting geometric racing-line optimization...")
    print("Number of points:", len(original_waypoints))
    print("Downsample:", DOWNSAMPLE)

    result = minimize(
        curvature_cost,
        initial_offsets,
        args=(center_points, normals),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 500,
            "ftol": 1e-6,
            "disp": True,
        },
    )

    if not result.success:
        print("\nWARNING: Optimization did not fully converge.")
        print(result.message)

    offsets = result.x

    optimized_points_2d = center_points[:, :2] + offsets[:, None] * normals
    optimized_points = center_points.copy()
    optimized_points[:, :2] = optimized_points_2d

    optimized_waypoints = create_optimized_waypoints(
        original_waypoints,
        optimized_points,
    )

    save_waypoints(OUTPUT_FILE, optimized_waypoints)

    print("\nDone.")
    print("Now run drawOptimizedWaypoints.py to visualize it.")


if __name__ == "__main__":
    main()
