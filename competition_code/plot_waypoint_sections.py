import os
import numpy as np
import matplotlib.pyplot as plt
import roar_py_interface


# ============================================================
# Ayub Section Visualizer
#
# Purpose:
#   1. Load your custom racing-line waypoints
#   2. Use the same physical section locations as the other team
#   3. Find the closest waypoint index for each section location
#   4. Plot each section in a different color
#   5. Print the section indices so you can copy them into submission.py
# ============================================================


# Change this if you want to plot a different waypoint file
WAYPOINT_FILE = "ayub_ipopt_raceline.npz"
# WAYPOINT_FILE = "ayub_ipopt_raceline_v2.npz"

# Same section locations used by the other team
SECTION_LOCATIONS = [
    [-278, 372],     # Section 0
    [64, 890],       # Section 1
    [511, 1037],     # Section 2
    [762, 908],      # Section 3
    [198, 307],      # Section 4
    [-11, 60],       # Section 5
    [-85, -339],     # Section 6
    [-210, -1060],   # Section 7
    [-318, -991],    # Section 8
    [-352, -119],    # Section 9
]


def get_base_path():
    return os.path.abspath(os.path.dirname(__file__))


def load_waypoints():
    waypoint_path = os.path.join(
        get_base_path(),
        "waypoints",
        WAYPOINT_FILE,
    )

    print("Loading waypoints from:")
    print(waypoint_path)

    waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
        np.load(waypoint_path)
    )

    print(f"Loaded {len(waypoints)} waypoints\n")
    return waypoints


def find_closest_index(location, waypoints):
    location = np.array(location)

    lowest_dist = 1e9
    closest_index = 0

    for i, waypoint in enumerate(waypoints):
        dist = np.linalg.norm(location[:2] - waypoint.location[:2])

        if dist < lowest_dist:
            lowest_dist = dist
            closest_index = i

    return closest_index, lowest_dist


def get_section_indices(waypoints):
    section_indices = []

    print("Section index mapping:")
    for i, loc in enumerate(SECTION_LOCATIONS):
        idx, dist = find_closest_index(loc, waypoints)
        section_indices.append(idx)
        print(f"Section {i}: location={loc}, closest_idx={idx}, distance={dist:.2f} m")

    return section_indices


def plot_sections(waypoints, section_indices):
    xy = np.array([wp.location[:2] for wp in waypoints])

    plt.figure(figsize=(12, 12))
    plt.title(f"Waypoint Sections: {WAYPOINT_FILE}")

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    for i in range(len(section_indices)):
        start_idx = section_indices[i]
        end_idx = section_indices[(i + 1) % len(section_indices)]

        if end_idx > start_idx:
            section_xy = xy[start_idx:end_idx + 1]
        else:
            section_xy = np.vstack((xy[start_idx:], xy[:end_idx + 1]))

        plt.plot(
            section_xy[:, 0],
            section_xy[:, 1],
            ".-",
            markersize=2,
            linewidth=1.2,
            color=colors[i % len(colors)],
            label=f"Section {i}",
        )

        plt.scatter(
            xy[start_idx, 0],
            xy[start_idx, 1],
            s=80,
            color=colors[i % len(colors)],
            edgecolors="black",
            zorder=5,
        )

        plt.text(
            xy[start_idx, 0],
            xy[start_idx, 1],
            f" S{i}\n idx {start_idx}",
            fontsize=9,
            weight="bold",
        )

    plt.scatter(
        xy[0, 0],
        xy[0, 1],
        s=120,
        color="black",
        marker="*",
        label="Waypoint 0",
        zorder=10,
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best")
    plt.tight_layout()

    output_name = f"section_plot_{WAYPOINT_FILE.replace('.npz', '')}.png"
    output_path = os.path.join(get_base_path(), output_name)

    plt.savefig(output_path, dpi=300)
    print(f"\nSaved section plot to:")
    print(output_path)

    plt.show()


def main():
    waypoints = load_waypoints()
    section_indices = get_section_indices(waypoints)

    print("\nCopy this into submission.py if needed:")
    print("self.section_indices = [")
    for idx in section_indices:
        print(f"    {idx},")
    print("]")

    plot_sections(waypoints, section_indices)


if __name__ == "__main__":
    main()
