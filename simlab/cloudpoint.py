#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2


class ImageToPlaneCloud(Node):
    def __init__(self):
        super().__init__('image_to_plane_cloud')

        # Parameters
        self.declare_parameter('image_topic', '/alpha/image_raw')
        self.declare_parameter('cloud_topic', '/alpha/points_plane')
        self.declare_parameter('z0', 1.0)          # plane depth in meters
        self.declare_parameter('hfov_deg', 80.0)   # horizontal FOV in degrees
        self.declare_parameter('vfov_deg', 64.0)   # vertical FOV in degrees
        self.declare_parameter('stride', 2)        # subsample factor, 1 means all pixels
        self.declare_parameter('frame_id', 'camera_optical_frame')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.z0 = float(self.get_parameter('z0').value)
        self.hfov = math.radians(float(self.get_parameter('hfov_deg').value))
        self.vfov = math.radians(float(self.get_parameter('vfov_deg').value))
        self.stride = int(self.get_parameter('stride').value)
        self.default_frame = self.get_parameter('frame_id').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.image_topic, self.cb, 10)
        self.pub = self.create_publisher(PointCloud2, self.cloud_topic, 10)

        self.get_logger().info(
            f'ImageToPlaneCloud started, topic={self.image_topic}, FOV=({math.degrees(self.hfov):.1f}h, {math.degrees(self.vfov):.1f}v), z0={self.z0} m'
        )

    def _to_rgb8(self, msg: Image) -> np.ndarray:
        """
        Convert any incoming Image to RGB8 numpy array.
        Handles common encodings: rgb8, bgr8, mono8, yuyv, uyvy, and passthrough paths.
        """
        enc = msg.encoding.lower()
        try:
            if enc == 'rgb8':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            elif enc == 'bgr8':
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif enc in ('mono8', '8uc1'):
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif enc in ('yuyv', 'yuv422', 'yuv422_yuy2', 'yuv422_yuyv', 'yuy2'):
                yuyv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                img = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
            elif enc in ('uyvy', 'yuv422_uyvy'):
                uyvy = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                img = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
            else:
                # Fallback
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed, {e}, trying naive RGB8')
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        return img

    def cb(self, msg: Image):
        img = self._to_rgb8(msg)
        h, w, _ = img.shape

        # Compute fx, fy from FOV and resolution
        # fx = w / (2 tan(hfov/2)), fy = h / (2 tan(vfov/2))
        fx = w / (2.0 * math.tan(self.hfov * 0.5))
        fy = h / (2.0 * math.tan(self.vfov * 0.5))
        cx, cy = w * 0.5, h * 0.5

        # Subsample grid
        ys = np.arange(0, h, self.stride, dtype=np.int32)
        xs = np.arange(0, w, self.stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.reshape(-1).astype(np.float32)
        grid_y = grid_y.reshape(-1).astype(np.float32)

        # Back project to a plane at z0
        z = np.full_like(grid_x, self.z0, dtype=np.float32)
        x = (grid_x - cx) * z / fx
        y = (grid_y - cy) * z / fy
        points_xyz = np.column_stack((x, y, z)).astype(np.float32)

        # Sample colors in the same pattern
        sampled = img[ys[:, None], xs[None, :]].reshape(-1, 3)
        colors_0to1 = sampled.astype(np.float32) / 255.0

        # Publish
        cloud_msg = self._make_cloud(
            stamp=msg.header.stamp,
            frame_id=msg.header.frame_id or self.default_frame,
            points_xyz=points_xyz,
            colors_0to1=colors_0to1
        )
        self.pub.publish(cloud_msg)

    @staticmethod
    def _make_cloud(stamp, frame_id, points_xyz, colors_0to1):
        # Pack colors into a single float32 rgb field
        c = np.clip(colors_0to1 * 255.0, 0, 255).astype(np.uint32)
        rgb_uint32 = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        rgb_float32 = rgb_uint32.view(np.float32)

        cloud = np.column_stack([points_xyz, rgb_float32]).astype(np.float32)

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return PointCloud2(
            header=header,
            height=1,
            width=cloud.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=16,
            row_step=16 * cloud.shape[0],
            is_dense=True,
            data=cloud.tobytes()
        )


def main():
    rclpy.init()
    node = ImageToPlaneCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
