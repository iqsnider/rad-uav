import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_point_cloud(n_points: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generate an arbitrary 3D point cloud simulating a LiDAR scan.

    The cloud is composed of:
      - A ground plane (flat cluster near z=0)
      - A vertical wall (cluster along one side)
      - A few object blobs (spherical clusters at various heights)

    Returns:
        points: np.ndarray of shape (N, 3) with columns [x, y, z]
                in the LiDAR sensor frame.
    """
    rng = np.random.default_rng(seed)

    parts = []

    # --- Ground plane ---
    n_ground = n_points // 3
    x = rng.uniform(-5, 5, n_ground)
    y = rng.uniform(-5, 5, n_ground)
    z = rng.normal(0.0, 0.05, n_ground)          # slight noise around z=0
    parts.append(np.column_stack([x, y, z]))

    # --- Vertical wall at x ≈ 4 ---
    n_wall = n_points // 4
    x = rng.normal(4.0, 0.05, n_wall)
    y = rng.uniform(-3, 3, n_wall)
    z = rng.uniform(0, 3, n_wall)
    parts.append(np.column_stack([x, y, z]))

    # --- Object blobs ---
    blob_centers = [(-2.0,  1.5, 0.5),
                    (1.0, -2.0, 0.8),
                    (0.5,  3.0, 1.2)]
    n_blob = (n_points - n_ground - n_wall) // len(blob_centers)
    for cx, cy, cz in blob_centers:
        blob = rng.normal(loc=[cx, cy, cz], scale=[
                          0.2, 0.2, 0.25], size=(n_blob, 3))
        parts.append(blob)

    points = np.vstack(parts)
    return points


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from roll/pitch/yaw (in radians).
    Convention: ZYX extrinsic (yaw → pitch → roll).

    This is the function you will use when applying LiDAR orientation data.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0, 1]])

    Ry = np.array([[cp, 0, sp],
                   [0, 1,  0],
                   [-sp, 0, cp]])

    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rx @ Ry @ Rz


def transform_point_cloud(
    points: np.ndarray,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply a rigid-body transform (rotation + translation) to the point cloud.

    Args:
        points:      (N, 3) array in the sensor frame.
        roll:        rotation around X axis [rad]
        pitch:       rotation around Y axis [rad]
        yaw:         rotation around Z axis [rad]
        translation: (3,) translation vector [x, y, z]

    Returns:
        transformed: (N, 3) array in the world frame.
    """
    R = rotation_matrix(roll, pitch, yaw)
    if translation is None:
        translation = np.zeros(3)

    # points @ R.T  is equivalent to  (R @ point.T).T  for each point
    transformed = points @ R.T + translation
    return transformed


def plot_point_cloud(
    points: np.ndarray,
    title: str = "LiDAR Point Cloud",
    color_by: str = "z",
) -> None:
    """
    Visualise a point cloud with matplotlib.

    Args:
        points:   (N, 3) array.
        title:    figure title.
        color_by: axis to use for colour mapping ('x', 'y', or 'z').
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    c_idx = axis_map.get(color_by, 2)
    colors = points[:, c_idx]

    fig = plt.figure(figsize=(10, 7))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap="plasma",
        s=1.5,
        alpha=0.6,
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label(f"{color_by.upper()} value", fontsize=9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_box_aspect([1, 1, 0.5])

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate raw point cloud (sensor frame)
    points = generate_point_cloud(n_points=20000)
    print(f"Generated point cloud: {points.shape[0]} points")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # 2. Plot the raw cloud
    plot_point_cloud(points, title="Raw LiDAR Point Cloud (sensor frame)")

    # 3. Example transform — replace these values with real IMU/orientation data
    roll_rad = np.deg2rad(5)     # e.g. from IMU roll
    pitch_rad = np.deg2rad(-3)    # e.g. from IMU pitch
    yaw_rad = np.deg2rad(45)    # e.g. from compass / yaw
    t = np.array([1.0, 0.5, 1.2])  # sensor position in world frame

    transformed = transform_point_cloud(
        points,
        roll=roll_rad,
        pitch=pitch_rad,
        yaw=yaw_rad,
        translation=t,
    )

    # 4. Plot the transformed cloud
    plot_point_cloud(
        transformed,
        title="Transformed Point Cloud (world frame)\n"
              f"roll={np.rad2deg(roll_rad):.1f}°  "
              f"pitch={np.rad2deg(pitch_rad):.1f}°  "
              f"yaw={np.rad2deg(yaw_rad):.1f}°  "
              f"t={t}",
    )
