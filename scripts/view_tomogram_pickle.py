#!/usr/bin/env python3
"""
Publish a PCT-compatible tomogram pickle to ROS 2 topics for RViz viewing.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
VENV_DIR = REPO_DIR / ".venv-rtabmap"


def add_venv_site_packages() -> None:
    if not VENV_DIR.is_dir():
        return
    for candidate in sorted(VENV_DIR.glob("lib/python*/site-packages")):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def import_open3d():
    try:
        import open3d as o3d

        return o3d
    except ModuleNotFoundError:
        add_venv_site_packages()
        import open3d as o3d

        return o3d


def grid_points_xyzi(resolution: float, dim_x: int, dim_y: int):
    index_proto = np.zeros((dim_x * dim_y, 2), dtype=np.int32)
    lx = np.linspace(0, dim_x - 1, dim_x, dtype=np.int32)
    ly = np.linspace(0, dim_y - 1, dim_y, dtype=np.int32)
    ix, iy = np.meshgrid(lx, ly)
    index_proto[:, 0] = ix.flatten()
    index_proto[:, 1] = iy.flatten()

    point_proto = np.zeros((dim_x * dim_y, 4), dtype=np.float32)
    point_proto[:, :2] = index_proto[:, :2].astype(np.float32, copy=True)
    point_proto[:, 0] -= 0.5 * dim_x
    point_proto[:, 1] -= 0.5 * dim_y
    point_proto[:, :2] *= resolution
    point_proto[:, 3] = 1.0
    return index_proto, point_proto


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"data", "resolution", "center", "slice_h0", "slice_dh"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Missing pickle keys: {sorted(missing)}")
    return payload


def load_optional_pcd(path: Path | None):
    if path is None or not path.is_file():
        return None
    o3d = import_open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        return None
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return None
    points = points[:, :3]
    result = np.zeros((points.shape[0], 4), dtype=np.float32)
    result[:, :3] = points
    result[:, 3] = points[:, 2]
    return result


def build_layer_points(index_proto, point_proto, center, layers, colors=None):
    messages = []
    for i in range(layers.shape[0]):
        layer_points = point_proto.copy()
        layer_points[:, :2] += center
        layer_points[:, 2] = layers[i, index_proto[:, 0], index_proto[:, 1]]
        if colors is not None:
            layer_points[:, 3] = colors[i, index_proto[:, 0], index_proto[:, 1]]
        else:
            layer_points[:, 3] = 1.0
        valid = layer_points[~np.isnan(layer_points).any(axis=-1)]
        messages.append(valid.astype(np.float32, copy=False))
    return messages


def build_tomogram_points(index_proto, point_proto, center, layers_g, layers_t, slice_dh):
    vis_g = layers_g.copy()
    vis_t = layers_t.copy()
    layer_points = point_proto.copy()
    layer_points[:, :2] += center
    global_points = []

    n_slice = layers_g.shape[0]
    for i in range(max(0, n_slice - 1)):
        mask_h = (vis_g[i + 1] - vis_g[i]) < slice_dh
        vis_g[i, mask_h] = np.nan
        vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1, mask_h])
        layer_points[:, 2] = vis_g[i, index_proto[:, 0], index_proto[:, 1]]
        layer_points[:, 3] = vis_t[i, index_proto[:, 0], index_proto[:, 1]]
        valid = layer_points[~np.isnan(layer_points).any(axis=-1)]
        if valid.size:
            global_points.append(valid.astype(np.float32, copy=False))

    layer_points[:, 2] = vis_g[-1, index_proto[:, 0], index_proto[:, 1]]
    layer_points[:, 3] = vis_t[-1, index_proto[:, 0], index_proto[:, 1]]
    valid = layer_points[~np.isnan(layer_points).any(axis=-1)]
    if valid.size:
        global_points.append(valid.astype(np.float32, copy=False))

    if global_points:
        return np.concatenate(global_points, axis=0)
    return np.zeros((0, 4), dtype=np.float32)

def create_cloud(points_xyzi, header, point_cloud2, point_field):
    fields = [
        point_field(name="x", offset=0, datatype=point_field.FLOAT32, count=1),
        point_field(name="y", offset=4, datatype=point_field.FLOAT32, count=1),
        point_field(name="z", offset=8, datatype=point_field.FLOAT32, count=1),
        point_field(name="intensity", offset=12, datatype=point_field.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, points_xyzi.tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a tomogram pickle to ROS 2 PointCloud2 topics.")
    parser.add_argument("--pickle", required=True, help="Path to a PCT-compatible tomogram pickle.")
    parser.add_argument("--pcd", help="Optional companion PCD path for /global_points.")
    parser.add_argument("--frame", default="map", help="Frame id for published messages.")
    args = parser.parse_args()

    pickle_path = Path(args.pickle).expanduser().resolve()
    if not pickle_path.is_file():
        print(f"Pickle not found: {pickle_path}", file=sys.stderr)
        return 1

    if args.pcd:
        pcd_path = Path(args.pcd).expanduser().resolve()
    else:
        pcd_path = pickle_path.with_suffix(".pcd")

    payload = load_pickle(pickle_path)
    data = np.asarray(payload["data"], dtype=np.float32)
    if data.ndim != 4 or data.shape[0] != 5:
        print(f"Unexpected tomogram data shape: {data.shape}", file=sys.stderr)
        return 1

    layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c = data
    _ = trav_grad_x, trav_grad_y
    resolution = float(payload["resolution"])
    center = np.asarray(payload["center"], dtype=np.float32)
    slice_dh = float(payload["slice_dh"])

    dim_x = layers_g.shape[1]
    dim_y = layers_g.shape[2]
    index_proto, point_proto = grid_points_xyzi(resolution, dim_x, dim_y)
    tomogram_points = build_tomogram_points(index_proto, point_proto, center, layers_g, layers_t, slice_dh)
    layer_g_points = build_layer_points(index_proto, point_proto, center, layers_g, layers_t)
    layer_c_points = build_layer_points(index_proto, point_proto, center, layers_c, None)
    global_points = load_optional_pcd(pcd_path)
    if global_points is None:
        global_points = tomogram_points.copy()

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2

    class TomogramViewer(Node):
        def __init__(self):
            super().__init__("rtabmap_tomogram_viewer")
            qos = QoSProfile(depth=1)
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            qos.reliability = ReliabilityPolicy.RELIABLE

            self.global_pub = self.create_publisher(PointCloud2, "/global_points", qos)
            self.tomogram_pub = self.create_publisher(PointCloud2, "/tomogram", qos)
            self.layer_g_pubs = [
                self.create_publisher(PointCloud2, f"/layer_G_{i}", qos)
                for i, points in enumerate(layer_g_points)
            ]
            self.layer_c_pubs = [
                self.create_publisher(PointCloud2, f"/layer_C_{i}", qos)
                for i, points in enumerate(layer_c_points)
            ]

            self.global_msg = create_cloud(global_points, self._header(), point_cloud2, PointField)
            self.tomogram_msg = create_cloud(tomogram_points, self._header(), point_cloud2, PointField)
            self.layer_g_msgs = [
                create_cloud(points, self._header(), point_cloud2, PointField) for points in layer_g_points
            ]
            self.layer_c_msgs = [
                create_cloud(points, self._header(), point_cloud2, PointField) for points in layer_c_points
            ]
            self.publish_once()

        def _header(self):
            header = Header()
            header.frame_id = args.frame
            header.stamp = self.get_clock().now().to_msg()
            return header

        def publish_once(self):
            self.global_msg.header = self._header()
            self.tomogram_msg.header = self._header()
            self.global_pub.publish(self.global_msg)
            self.tomogram_pub.publish(self.tomogram_msg)
            for i, pub in enumerate(self.layer_g_pubs):
                self.layer_g_msgs[i].header = self._header()
                pub.publish(self.layer_g_msgs[i])
            for i, pub in enumerate(self.layer_c_pubs):
                self.layer_c_msgs[i].header = self._header()
                pub.publish(self.layer_c_msgs[i])

    rclpy.init()
    node = TomogramViewer()
    node.get_logger().info(f"Publishing tomogram from {pickle_path}")
    if pcd_path.is_file():
        node.get_logger().info(f"Publishing companion cloud from {pcd_path}")
    else:
        node.get_logger().info("No companion PCD found, reusing tomogram cloud for /global_points")
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
