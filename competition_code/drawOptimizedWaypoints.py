import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
import os
from progress.bar import IncrementalBar
import transforms3d as tr3d


print("\nLoading Waypoints\n")

base_path = os.path.abspath(os.path.dirname(__file__))

optimized = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(os.path.join(base_path, "waypoints", "ayub_monza_spicy_line.npz"))
)

track = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(os.path.join(base_path, "waypoints", "Monza Original Waypoints.npz"))
)

totalPoints = len(optimized) + len(track)
progressBar = IncrementalBar("Plotting points", max=totalPoints)

plt.figure(figsize=(11, 11))
plt.axis((-1100, 1100, -1100, 1100))
plt.tight_layout()

# Plot original track representation in black
for waypoint in track[:] if track is not None else []:
    rep_line = waypoint.line_representation
    rep_line = np.asarray(rep_line)

    waypoint_heading = tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array(
        [1, 0, 0]
    )

    plt.plot(rep_line[:, 0], rep_line[:, 1], "k", linewidth=2)

    plt.arrow(
        waypoint.location[0],
        waypoint.location[1],
        waypoint_heading[0] * 1,
        waypoint_heading[1] * 1,
        width=0.5,
        color="r",
    )

    progressBar.next()

# Plot optimized racing line
for waypoint in optimized:
    plt.plot(waypoint.location[0], waypoint.location[1], "bo", markersize=2)
    progressBar.next()

progressBar.finish()
print()
plt.title("Ayub Optimized Racing Line")
plt.show()
