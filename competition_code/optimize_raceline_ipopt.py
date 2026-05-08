import os
import math
import numpy as np
import casadi as ca
import roar_py_interface


# ============================================================
# Ayub IPOPT Racing Line Optimizer
#
# Input:
#   waypoints/Monza Original Waypoints.npz
#
# Output:
#   waypoints/ayub_ipopt_raceline.npz
#
# This is a geometric racing-line optimizer.
# It does NOT need mass, yaw inertia, tire model, or wheelbase.
#
# It optimizes lateral offsets from the centerline:
#
#   race_point_i = center_point_i + offset_i * normal_i
#
# Objective:
#   reduce path curvature / sharpness
#   reduce zig-zag
#   stay inside lane width
# ============================================================


INPUT_FILE = "Monza Original Waypoints.npz"
OUTPUT_FILE = "ayub_ipopt_raceline.npz"

# Higher = fewer points = faster optimization.
# Start with 10 or 15.
# Later try 6 or 5 for better quality.
DOWNSAMPLE = 1

# Keep away from track edge
TRACK_MARGIN = 3.3

# Cost weights
CURVATURE_WEIGHT = 1.0
OFFSET_SMOOTHNESS_WEIGHT = 1.0
OFFSET_MAGNITUDE_WEIGHT = 0.0


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

    print("\nSaved IPOPT racing line to:")
    print(path)


def compute_normals(points):
    """
    Compute normal direction at each centerline point.
    """
    normals = []
    n = len(points)

    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_next = points[(i + 1) % n]

        tangent = p_next[:2] - p_prev[:2]
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm < 1e-6:
            normal = np.array([0.0, 1.0])
        else:
            tangent = tangent / tangent_norm
            normal = np.array([-tangent[1], tangent[0]])

        normals.append(normal)

    return np.array(normals)


def build_optimized_waypoints(original_waypoints, optimized_points):
    """
    Convert optimized x,y points back into RoarPyWaypoint format.
    """
    optimized_waypoints = []
    n = len(optimized_points)

    for i in range(n):
        p = optimized_points[i]
        p_next = optimized_points[(i + 1) % n]

        dx = p_next[0] - p[0]
        dy = p_next[1] - p[1]

        yaw = math.atan2(dy, dx)

        location = np.array([
            p[0],
            p[1],
            original_waypoints[i].location[2],
        ])

        roll_pitch_yaw = np.array([0.0, 0.0, yaw])

        lane_width = original_waypoints[i].lane_width
        if lane_width is None or lane_width <= 0:
            lane_width = 12.0

        optimized_waypoints.append(
            roar_py_interface.RoarPyWaypoint(
                location=location,
                roll_pitch_yaw=roll_pitch_yaw,
                lane_width=lane_width,
            )
        )

    return optimized_waypoints


def optimize_with_ipopt(center_points, normals, lane_widths):
    n = center_points.shape[0]

    # Decision variable:
    # lateral offset from centerline at every waypoint
    offset = ca.MX.sym("offset", n)

    cost = 0

    # Build optimized x/y symbolic points
    x = []
    y = []

    for i in range(n):
        xi = center_points[i, 0] + offset[i] * normals[i, 0]
        yi = center_points[i, 1] + offset[i] * normals[i, 1]
        x.append(xi)
        y.append(yi)

    # Curvature-like cost:
    # penalize second difference of x/y
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n

        ddx = x[ip1] - 2 * x[i] + x[im1]
        ddy = y[ip1] - 2 * y[i] + y[im1]

        cost += CURVATURE_WEIGHT * (ddx**2 + ddy**2)

    # Offset smoothness cost:
    # prevents left-right-left zig-zag
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n

        offset_second_diff = offset[ip1] - 2 * offset[i] + offset[im1]
        cost += OFFSET_SMOOTHNESS_WEIGHT * offset_second_diff**2

    # Small penalty on using full edge all the time
    for i in range(n):
        cost += OFFSET_MAGNITUDE_WEIGHT * offset[i]**2

    # Bounds from lane width
    lower_bounds = []
    upper_bounds = []

    for lane_width in lane_widths:
        if lane_width is None or lane_width <= 0:
            lane_width = 12.0

        usable_half_width = max(0.5, lane_width / 2.0 - TRACK_MARGIN)

        lower_bounds.append(-usable_half_width)
        upper_bounds.append(usable_half_width)

    nlp = {
        "x": offset,
        "f": cost,
    }

    options = {
        "ipopt.print_level": 5,
        "print_time": True,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
    }

    solver = ca.nlpsol("solver", "ipopt", nlp, options)

    initial_guess = np.zeros(n)

    print("\nStarting IPOPT optimization...")
    print("Optimization points:", n)
    print("Downsample:", DOWNSAMPLE)

    solution = solver(
        x0=initial_guess,
        lbx=np.array(lower_bounds),
        ubx=np.array(upper_bounds),
    )

    optimized_offsets = np.array(solution["x"]).flatten()

    return optimized_offsets


def main():
    original_waypoints_full = load_waypoints(INPUT_FILE)

    original_waypoints = original_waypoints_full[::DOWNSAMPLE]

    center_points = np.array([
        wp.location for wp in original_waypoints
    ])

    lane_widths = np.array([
        wp.lane_width if wp.lane_width is not None and wp.lane_width > 0 else 12.0
        for wp in original_waypoints
    ])

    normals = compute_normals(center_points)

    offsets = optimize_with_ipopt(center_points, normals, lane_widths)

    optimized_points = center_points.copy()
    optimized_points[:, :2] = center_points[:, :2] + offsets[:, None] * normals

    optimized_waypoints = build_optimized_waypoints(
        original_waypoints,
        optimized_points,
    )

    save_waypoints(OUTPUT_FILE, optimized_waypoints)

    print("\nDone.")
    print("Now run drawOptimizedWaypoints.py.")
    print("Make sure drawOptimizedWaypoints.py loads:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
