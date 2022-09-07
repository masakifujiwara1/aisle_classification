#!/usr/bin/env python3
from __future__ import print_function
from xmlrpc.client import FastParser
from nav_msgs.msg import Odometry
import tf
import sys
import copy
import time
import os
import csv
from std_srvs.srv import SetBool, SetBoolResponse
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from std_msgs.msg import Int8
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Twist
from skimage.transform import resize
from aisle_class_net import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import roslib
# roslib.load_manifest('nav_cloning')

#(直進, 角, 三叉路)
INPUT = 64  # 64
COOL_TIME = 120
AISLE_DATA = 30
MINIMUM_ELEMNTS = 20


class aisle_class_node:
    def __init__(self):
        rospy.init_node('aisle_class_node', anonymous=True)
        self.action_num = 3
        self.dl = deep_learning(n_out=self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber(
            "/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber(
            "/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.cmd_dir_sub = rospy.Subscriber(
            "/cmd_dir", Int8MultiArray, self.callback_cmd, queue_size=1)
        self.pose_sub = rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.episode = 0
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_left_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_right_image = np.zeros((480, 640, 3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.save_path = roslib.packages.get_pkg_dir(
            'aisle_classification') + '/data/models/'
        self.load_path = roslib.packages.get_pkg_dir(
            'aisle_classification') + '/data/models/8000step_input64/model_gpu.pt'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.aisle_class = (1, 0, 0)
        self.start_time_s = rospy.get_time()
        self.pos_aisle_x = 0
        self.pos_aisle_y = 0
        self.correct_count = 0
        self.aisle_status = '道なり'
        self.aisle_flag_corner = False
        self.aisle_flag_sansaro = False
        self.aisle_cool_count = COOL_TIME
        self.aisle_list = []
        self.aisle_pose_corner = np.array([
            [[-11, -6.5], [-7.3, -1.4]],
            [[-3.6, -6.6], [0.7, -1.1]],
            [[-10, 27], [-6.7, 31]],
            [[-2.3, 27], [1.4, 30]]
        ])
        self.aisle_pose_sansaro = np.array([
            [[-11.3, 4.2], [-6.8, 11]],
            [[-3.5, 5.1], [0.85, 11.1]]
        ])

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion(
            (rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        self.pos_aisle_x = data.pose.pose.position.x
        self.pos_aisle_y = data.pose.pose.position.y

        if self.episode >= 1:
            self.check_area()

        # for pose in self.path_pose.poses:
        #     path = pose.pose.position
        #     distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
        #     distance_list.append(distance)

        # if distance_list:
        #     self.min_distance = min(distance_list)

    def callback_cmd(self, data):
        # self.aisle_class = data.data
        pass

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def check_area(self):
        for i, ((x1, y1), (x2, y2)) in enumerate(self.aisle_pose_corner):
            if (x1 <= self.pos_aisle_x <= x2) and (y1 <= self.pos_aisle_y <= y2):
                self.aisle_flag_corner = True
                self.aisle_class = (0, 1, 0)
                # print("角")
                self.aisle_status = "角"

        for i, ((x1, y1), (x2, y2)) in enumerate(self.aisle_pose_sansaro):
            if (x1 <= self.pos_aisle_x <= x2) and (y1 <= self.pos_aisle_y <= y2):
                self.aisle_flag_sansaro = True
                self.aisle_class = (0, 0, 1)
                # print("三叉路")
                self.aisle_status = "三叉路"

        if (not self.aisle_flag_corner) and (not self.aisle_flag_sansaro):
            self.aisle_class = (1, 0, 0)
            # print("道なり")
            self.aisle_status = "道なり"

        self.aisle_flag_corner = False
        self.aisle_flag_sansaro = False

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        img = resize(self.cv_image, (INPUT, INPUT), mode='constant')
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r, g, b])

        img_left = resize(self.cv_left_image, (128, 128), mode='constant')
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r, g, b])

        img_right = resize(self.cv_right_image, (128, 128), mode='constant')
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r, g, b])
        cmd_dir = np.asanyarray(self.aisle_class)
        ros_time = str(rospy.Time.now())

        if self.episode == 0:
            self.learning = False
            self.dl.load(self.load_path)

        if self.episode == 8000:
            self.learning = False
            self.dl.save(self.save_path)
            # self.dl.load(self.load_path)

            # not test mode
            os.system('killall roslaunch')
            sys.exit()

        if self.episode == 12000:
            os.system('killall roslaunch')
            sys.exit()

        if self.learning:
            # target_action = self.action
            # distance = self.min_distance

            class_, class_1, class_2, loss = self.dl.detect_and_trains(
                img, self.aisle_class)
            print(self.episode, class_, class_1, class_2)
            # class_, loss_r = self.de.detect_and_train(img_right, self.aisle_class)
            # class_, loss_l = self.de.detect_and_train(img_left, self.aisle_class)

            # end mode

            # print(str(self.episode) + ", training, loss: " + str(loss) + ", aisle_class: " +
            #       str(self.aisle_class) + ", training_aisle_class: " + str(class_))
            self.episode += 1
            # line = [str(self.episode), "training", str(loss), str(angle_error), str(
            #     distance), str(self.pos_x), str(self.pos_y), str(self.pos_the), str(cmd_dir)]
            # with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     writer.writerow(line)
            # self.vel.linear.x = 0.2
            # self.vel.angular.z = target_action
            # self.nav_pub.publish(self.vel)

        else:
            class_ = self.dl.detect(img)
            # print(class_)
            max_index = class_[0].index(max(class_[0]))

            if max_index == 0:
                dict_class = "道なり"
            elif max_index == 1:
                dict_class = "角"
            else:
                dict_class = "三叉路"

            self.aisle_list.append(dict_class)

            if len(self.aisle_list) > AISLE_DATA:
                del self.aisle_list[0]

            print(self.aisle_list.count('角'), self.aisle_list.count(
                '三叉路'), self.aisle_cool_count)

            if self.aisle_cool_count > COOL_TIME:
                if self.aisle_list.count('角') > MINIMUM_ELEMNTS:
                    print("detect 角")
                    self.aisle_cool_count = 0
                elif self.aisle_list.count('三叉路') > MINIMUM_ELEMNTS:
                    print("detect 三叉路")
                    self.aisle_cool_count = 0

            if dict_class == self.aisle_status:
                self.correct_count += 1

            per_ = self.correct_count/(self.episode + 1e-4) * 100
            if per_ > 100:
                per_ = 100

            # distance = self.min_distance
            # print(str(self.episode) + ", test, class:" +
            #       str(max_index) + ", currnt_class: " + str(cmd_dir) + 'actually_class:' + str(class_))
            print(str(self.episode) + ", test, dict_class:" +
                  str(dict_class) + ", correct_class: " + self.aisle_status + ", acc_per: " + str(per_))

            self.episode += 1
            self.aisle_cool_count += 1
            # angle_error = abs(self.action - target_action)
            # line = [str(self.episode), "test", "0", str(angle_error), str(distance), str(
            #     self.pos_x), str(self.pos_y), str(self.pos_the), str(cmd_dir)]
            # with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     writer.writerow(line)
            # self.vel.linear.x = 0.2
            # self.vel.angular.z = target_action
            # self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        # temp = copy.deepcopy(img_left)
        # cv2.imshow("Resized Left Image", temp)
        # temp = copy.deepcopy(img_right)
        # cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)


if __name__ == '__main__':
    rg = aisle_class_node()
    DURATION = 0.25
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
