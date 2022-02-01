#! /usr/bin/env python

from typing import Union

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from tf import transformations, TransformListener, TransformBroadcaster
from HoleFilling import HoleFillingFilter


class SceneGazeIntersectionProcessNode:

    def __init__(self):
        self._bridge = CvBridge()
        self._tf_listener = TransformListener()
        self._tf_publisher = TransformBroadcaster()
        self._cone_angle = rospy.get_param("cone_angle_degree", default=np.deg2rad(4))
        self._cone_threshold = np.sin(self._cone_angle)
        self._sample_points = rospy.get_param("sample_points", default=5)
        self._sample_j = rospy.get_param("sample_j", default=1000)
        self._depth_resize = rospy.get_param("depth_resize", default=0.2)
        self._publish_tf = rospy.get_param("publish_tf", default=True)
        fill_hole = rospy.get_param("depth_hole_fill", default=True)
        self._hole_filler = HoleFillingFilter(radius=2) if fill_hole is not None else lambda x: x

        print("Waiting for Camera Info msg", end="")
        cam_info = rospy.wait_for_message("/camera_info", CameraInfo, timeout=None)
        self._camera = PinholeCameraModel()
        self._camera.fromCameraInfo(cam_info)
        print("...Received")

        self._gaze_tf = rospy.get_param("gaze_pose", default="head_camera/world_gaze0")
        self._head_tf = rospy.get_param("head_pose", default="head_camera/head_pose0")
        self._world_tf = rospy.get_param("camera_world_frame", default="zed_left_camera_optical_frame")

        self._sub = rospy.Subscriber("/image", Image, self.process_image, buff_size=2 ** 24, queue_size=3)
        self._pub = rospy.Publisher("/gaze_scene_intersection", PoseStamped)

    def _cast_ray(self, width, height, head_pose, gaze_direction):
        x, y = np.meshgrid(range(width), range(height))

        vec = np.array([x - head_pose[0], y - head_pose[1]])
        vec = vec / np.linalg.norm(vec, axis=0)
        gaze_direction = np.array(gaze_direction)
        gaze_direction /= np.linalg.norm(gaze_direction)

        # this is magic; facilitates a (2, N, M) x (2, 1)
        shape = np.swapaxes(vec, vec.ndim - 1, 0).shape
        gaze_direction_brc = np.broadcast_to(gaze_direction, shape)
        gaze_direction_brc = np.swapaxes(gaze_direction_brc, vec.ndim - 1, 0)

        sim = vec * gaze_direction_brc
        sim = np.clip(np.sum(sim, axis=0), a_max=1, a_min=-1)
        direc = np.arccos(sim)
        direc = np.maximum(1 - (12 * direc / np.pi), 0)
        return direc

    @staticmethod
    def _rotate_vector(angle, dir1, dir2):
        x = (np.cos(angle) * (dir2[0] - dir1[0])) - (np.sin(angle) * (dir2[1] - dir1[1]))
        y = (np.sin(angle) * (dir2[0] - dir1[0])) + (np.cos(angle) * (dir2[1] - dir1[1]))

        return dir1 + np.array([x, y])

    @staticmethod
    def _gaus2d(width, height, centre=(0, 0), spread=(1, 1)):
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        x, y = np.meshgrid(x, y)
        z = 1.0 / (2.0 * np.pi * spread[0] * spread[1]) * np.exp(-((x - centre[0]) ** 2.0 / (2.0 * spread[0] ** 2.0) + (y - centre[1]) ** 2.0 / (2.0 * spread[1] ** 2.0)))
        return z

    @staticmethod
    def _point_distance_to_line(point, start, end):
        start_vec = np.array(start)
        end_vec = np.array(end)
        stare_dist = np.linalg.norm(start_vec - end_vec)
        d = (end_vec - start_vec) / stare_dist
        v = point - end_vec
        t = np.dot(v, d)
        project_p = end_vec + t * d
        distance = np.linalg.norm(point - project_p)
        return distance

    def intersection_point(self, depth_img: np.ndarray, start_position_3d: np.ndarray, end_position_3d: np.ndarray) -> Union[tuple, None]:
        start_u, start_v = self._camera.project3dToPixel((start_position_3d[0], start_position_3d[1], start_position_3d[2]))
        end_u, end_v = self._camera.project3dToPixel((end_position_3d[0], end_position_3d[1], end_position_3d[2]))

        p1 = np.array([start_u, start_v])
        p2 = np.array([end_u, end_v])
        p3 = self._rotate_vector(self._cone_angle, p1, p2)
        p4 = self._rotate_vector(-self._cone_angle, p1, p2)

        # sampling_dist = self._cast_ray(width=depth_img.shape[1], height=depth_img.shape[0], head_pose=(start_u, start_v), gaze_direction=p2 - p1)

        triangle_cnt = np.array([p1, p3, p2, p4]).reshape((-1, 1, 2)).astype(np.int64)
        scratch = np.zeros_like(depth_img)
        scratch = cv2.drawContours(scratch, contours=[triangle_cnt], contourIdx=-1, color=255, thickness=-1)

        # gaze_dist = np.linalg.norm(p2 - p1)
        # empiric_dist = self._gaus2d(width=depth_img.shape[1], height=depth_img.shape[0], centre=(start_u, start_v), spread=(gaze_dist / 2.0, gaze_dist / 2.0))
        sampling_dist = scratch
        _sum = np.nansum(scratch)
        if np.allclose(_sum, 0):
            return None

        sampling_dist /= _sum

        choices_idx = np.random.choice(depth_img.shape[0] * depth_img.shape[1], size=self._sample_j, p=sampling_dist.ravel())

        possible_points = []
        for u, v in zip(*np.unravel_index(choices_idx, depth_img.shape)):
            depth = depth_img[u, v]

            if not np.isnan(depth) and not np.isinf(depth):
                proj_3d = self._camera.projectPixelTo3dRay((v, u))
                p = np.array([proj_3d[0], proj_3d[1], depth])

                p_line_distance = self._point_distance_to_line(p, start=start_position_3d, end=end_position_3d)
                p_head_distance = np.linalg.norm(p - start_position_3d)
                threshold = p_head_distance * self._cone_threshold

                if p_line_distance <= threshold:
                    possible_points.append(((p_line_distance, p_head_distance), (v, u)))

        # find the smallest distances
        possible_points.sort(key=lambda k: k[0][0])
        if len(possible_points) > 0:
            idx = np.minimum(len(possible_points), self._sample_points)
            sorted_possible_points = possible_points[:idx]
            sorted_possible_points.sort(key=lambda k: k[0][1])
            return sorted_possible_points[0][1]
        else:
            return None

    def _transform_pt(self, pt, source_frame, target_frame, time):
        msg = self._tf_listener.lookupTransform(target_frame=target_frame, source_frame=source_frame, time=time)
        t = msg.transform.translation
        r = msg.transform.rotation
        translation, rotation = [t.x, t.y, t.z], [r.x, r.y, r.z, r.w]
        mat44 = np.dot(transformations.translation_matrix(translation), transformations.quaternion_matrix(rotation))

        p2 = np.dot(mat44, np.array([pt[0], pt[1], pt[2], 1.0]))[:3]
        return p2

    def process_image(self, msg):

        depth_img_orig = self._bridge.imgmsg_to_cv2(msg)
        depth_img = cv2.resize(depth_img_orig, dsize=(0, 0), fx=self._depth_resize, fy=self._depth_resize)
        try:
            depth_img = self._hole_filler(depth_img)

            gaze_world_transform = self._tf_listener.lookupTransform(target_frame=self._world_tf, source_frame=self._gaze_tf, time=rospy.Time(0))  # latest time available
            gaze_start = gaze_world_transform.transform.translation
            gaze_end = self._transform_pt((250, 0, 0), source_frame=self._gaze_tf, target_frame=self._world_tf, time=rospy.Time(0))
            scene_intersection_px = self.intersection_point(depth_img=depth_img, start_position_3d=np.array([gaze_start.x, gaze_start.y, gaze_start.z]),
                                                            end_position_3d=np.array(gaze_end))

            if scene_intersection_px is not None:
                data = np.array(scene_intersection_px) * (1.0 / self._depth_resize)
            else:
                data = None

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            data = None
            pass

        if data is not None:
            msg = PoseStamped()
            msg.header.stamp = msg.header.stamp
            msg.header.frame_id = msg.header.frame_id
            msg.pose.position.x = data[0]
            msg.pose.position.y = data[1]
            msg.pose.position.z = data[2]
            msg.pose.orientation.x = 0
            msg.pose.orientation.y = 0
            msg.pose.orientation.z = 0
            msg.pose.orientation.w = 1

            if self._publish_tf:
                self._tf_publisher.sendTransform(translation=data, rotation=(0, 0, 0, 1), time=msg.header.stamp, child=self._gaze_tf + "/scene_intersection", parent=self._world_tf)

            self._pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("gaze_scene_intersection")
    node = SceneGazeIntersectionProcessNode()
    rospy.spin()
