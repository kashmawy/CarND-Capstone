#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
# from PIL import Image as PIL_Image
from light_classification.tl_classifier import TLClassifier
from light_classification.tl_classifier_detection import TLClassifierDetection
import tf
from tf import transformations as t
import cv2
import yaml
import waypoint_lib.helper as helper
import os.path
import message_filters
import pickle
# import datetime
import time
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3
dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

# Sim
# LIGHT_DETECT_DIST_MIN = 13 # m
# LIGHT_DETECT_DIST_MAX = 120 # m

# Site
# LIGHT_DETECT_DIST_MIN = 4 # m
# LIGHT_DETECT_DIST_MAX = 26 # m


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')


        self.pose = None

        self.waypoints = None
        self.camera_image = None
        self.camera_image_prev_seq = None
        self.lights = []
        self.nr = 0
        self.state = None

        self.cv_image_test = None


        # It should be False for final submission, because we want to use provided transform
        # rather than our own from pose
        self.use_pose_transform = True # rospy.get_param('~use_pose_transform') # mps


        self.light_detect_dist_min = rospy.get_param('~light_detect_dist_min', 4.) # m
        self.light_detect_dist_max = rospy.get_param('~light_detect_dist_max', 26.) # m
        rospy.loginfo('~light_detect_dist_min = {}'.format(self.light_detect_dist_min))
        rospy.loginfo('~light_detect_dist_max = {}'.format(self.light_detect_dist_max))




        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)


        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        # Subscribers for sim training data recordings
        # image_sub = message_filters.Subscriber('/image_color', Image)
        # pose_sub = message_filters.Subscriber('/current_pose', PoseStamped)
        # lights_sub = message_filters.Subscriber('/vehicle/traffic_lights', TrafficLightArray)

        # Uncomment this line to start collecting data
        # ts = message_filters.ApproximateTimeSynchronizer([image_sub, pose_sub, lights_sub], 10, 0.005)
        # ts.registerCallback(self.image_sync)

        # Porocess image for traffic lights
        # ts = message_filters.ApproximateTimeSynchronizer([image_sub, pose_sub], 10, 0.05)
        # ts.registerCallback(self.process_traffic_lights_sync)

        self.record_name = time.strftime("%Y%m%d%H%M%S") # , datetime.datetime.now()
        self.records = []



        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.image_test = rospy.Publisher('/image_test', Image, queue_size=1)

        self.bridge = CvBridge()

        # Get models param from launch file
        self.model_dir = rospy.get_param('~model_dir', './light_classification/detection_api/inference_models/faster_rcnn_multi_site') # path to the model
        self.consensus = rospy.get_param('~consensus', 1) # min num of lights detected in one type

        self.light_classifier = TLClassifierDetection(self.model_dir, consensus = self.consensus)

        # self.light_classifier = TLClassifier()
        # self.light_classifier = TLClassifierDetection('./light_classification/detection_api/inference_models/ssd_filtered_sim', consensus = 1) # ssd filtered model
        # self.light_classifier = TLClassifierDetection('./light_classification/detection_api/inference_models/faster_rcnn_multi_filtered', consensus = 1) # faster rcnn multi filtered model

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.record_cnt = 0
        self.sim_state = 5




        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        # rospy.loginfo('>>> got pose')

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # rospy.loginfo('>>> got waypoints')

    def traffic_cb(self, msg):
        self.lights = msg.lights
        # rospy.loginfo('>>> got traffic_lights')

    def process_traffic_lights_sync(self, image_msg, pose_msg):
        # rospy.loginfo('---- PROCESS TRAFFIC LIGHTS SYNC ------')
        # rospy.loginfo('use_pose_transform = {}'.format(self.use_pose_transform))
        self.has_image = True
        self.camera_image = image_msg
        self.pose = pose_msg

        light_wp, state = self.process_traffic_lights(image_msg, pose_msg)


        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

        # Debug output
        if self.cv_image_test is not None:
            cv2.putText(self.cv_image_test, 'Red Light WP: {}'.format(self.last_wp), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
            self.publish_image_test(self.cv_image_test)

            # save image test to folder
            '''
            class_folder = os.path.join('.', 'output_images', self.record_name, 'class')
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            class_img_fname = os.path.join(class_folder, 'classifier{}_{}_{}.jpg'.format(self.nr, self.sim_state, state))
            cv2.imwrite(class_img_fname, self.cv_image_test)
            self.nr = self.nr + 1
            '''

    # Used for recording sim training data ONLY
    def image_sync(self, image_msg, pose_msg, lights_msg):
        rospy.loginfo('---- IMAGE SYNC ------')

        lights_msg = lights_msg.lights

        # Select the closest waypoint from lights array which was received from /vehicle/traffic_lights topic
        # Skip if we already processed this image
        if (self.waypoints and image_msg.header.seq != self.camera_image_prev_seq):


            self.camera_image_prev_seq = image_msg.header.seq

            car_wp = helper.next_waypoint_idx(pose_msg, self.waypoints.waypoints)

            lights_wp = [helper.closest_waypoint_idx(l.pose, self.waypoints.waypoints) for l in lights_msg]
            lights_dists = [helper.wp_distance(car_wp, lwp, self.waypoints.waypoints) for lwp in lights_wp]
            closest_light = lights_dists.index(min(lights_dists))

            rospy.loginfo('--- car_wp = {}'.format(car_wp))
            rospy.loginfo('--- closest_light[{}] = {}, {}'.format(closest_light, lights_msg[closest_light].state, lights_dists[closest_light]))

            # light = lights_wp[closest_light]
            light_wp = lights_wp[closest_light]
            light = lights_msg[closest_light]

            # This we have only in simulator for testing
            state = lights_msg[closest_light].state
            rospy.loginfo('--- SIM: closest_light_wp = {}, state = {}'.format(light_wp, light.state))

            # Collect test data
            # self.record_camera_image(light, state)


            # Check that folder exists
            img_folder = os.path.join('.', 'output_images', self.record_name, 'imgs_full')
            output_folder = os.path.join('.', 'output_images', self.record_name)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            record_filename = os.path.join(output_folder, "{}.pickle".format(self.record_name))

            # rospy.loginfo("img_folder = {}".format(os.path.dirname(os.path.abspath(img_folder))))

            # car_wp = helper.next_waypoint_idx(pose_msg, self.waypoints.waypoints)
            # light_wp = helper.closest_waypoint_idx(light.pose, self.waypoints.waypoints)

            # Is light close enough?
            waypoints_num = len(self.waypoints.waypoints)
            light_dist = (light_wp - car_wp + waypoints_num) % waypoints_num
            light_dist_m = lights_dists[closest_light]

            # Save image
            if self.light_detect_dist_min < light_dist_m < self.light_detect_dist_max:

                self.record_cnt += 1

                rospy.loginfo('--- =============== saving image record_cnt = {} ....'.format(self.record_cnt))

                img_filename = os.path.join(output_folder, 'imgs_full', '{:06d}-{}-{}.png'.format(self.record_cnt, state, int(light_dist_m)))

                img_record = {}
                img_record['filename'] = img_filename
                img_record['pose_msg'] = pose_msg
                img_record['traffic_light'] = light
                img_record['light_dist_wp'] = light_dist
                img_record['state'] = state

                rospy.loginfo("--- saving: {}, dist = {}".format(img_filename, light_dist))
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

                '''
                # Crop Image
                # Crop project traffic light position and crop image
                height, width, channels = cv_image.shape
                pos = light.pose.pose.position
                x_up, y_up = self.project_to_image_plane(pos, 0.5, 1, pose_msg)
                x_down, y_down = self.project_to_image_plane(pos, -0.5, -1, pose_msg)
                rospy.loginfo('x_up = {}, y_up = {}, x_down = {}, y_down = {} === hw = {}:{}'.format(x_up, y_up, x_down, y_down, height, width))

                if x_up > width or y_down > height or x_up < 0 or y_down < 0:
                    return TrafficLight.UNKNOWN

                cpy = cv_image.copy()

                if x_down is None or x_up is None or y_up is None or y_down is None:
                    return TrafficLight.UNKNOWN
                if x_down - x_up < 20 or y_down - y_up < 40:
                    return TrafficLight.UNKNOWN

                output_crop = cpy[int(y_up):int(y_down), int(x_up):int(x_down)]
                # cv2.circle(cpy, ((x_up+x_down)/2, (y_up+y_down)/2), 10, (0, 255, 0), 2)
                '''

                rospy.loginfo('to file ...')
                cv2.imwrite(img_filename, cv_image) # output_crop

                rospy.loginfo('to topic ...')
                self.publish_image_test(cv_image)

                # rospy.loginfo('img_record = {}'.format(img_record))

                self.records.append(img_record)

                # Output to file every 1000 calls
                if self.record_cnt % 100 == 0:
                    with open(record_filename, 'wb') as rf:
                        pickle.dump(self.records, rf)
                        rospy.loginfo('SAVED TO FILE: {}'.format(record_filename))


            else:
                rospy.loginfo("--- light is far away: {:.2f} meters".format(light_dist_m))



    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.has_image = True
        self.camera_image = msg
        # rospy.loginfo('>>> got image')

        if self.pose:
            self.process_traffic_lights_sync(self.camera_image, self.pose)


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        vehicle = pose.position
        wayp = self.waypoints.waypoints
        distanceList = []

        for i in range(len(wayp)):
            distanceList.append(dl(vehicle, wayp[i].pose.pose.position))

        return distanceList.index(min(distanceList))


    def project_to_image_plane(self, point_in_world, offsetX, offsetY, pose = None):
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
        cx = image_width/2
        cy = image_height/2
        transT = None
        rotT = None

        # Use provided pose or if use_pose_transform is set use
        # self.pose
        if not pose and self.use_pose_transform:
            pose = self.pose


        if pose:
            # This code is used by image_sync because we want to use transform from current pose
            # or when we work with provided bag
            transT, rotT = helper.get_inverse_trans_rot(self.pose)
        else:
            # This code we need for site mode (and simulator)
            try:
                now = rospy.Time.now()
                self.listener.waitForTransform("/base_link",
                      "/world", now, rospy.Duration(1.0))
                (transT, rotT) = self.listener.lookupTransform("/base_link",
                      "/world", now)

            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                rospy.logerr("Failed to find camera to map transform")
                return None, None


        px = point_in_world.x
        py = point_in_world.y
        pz = point_in_world.z
        rpy = tf.transformations.euler_from_quaternion(rotT)

        yaw = rpy[2]

        point_cam = (px * math.cos(yaw) - py * math.sin(yaw),
                        px * math.sin(yaw) + py * math.cos(yaw),
                        pz)
        point_cam = [sum(x) for x in zip(point_cam, transT)]

        point_cam[1] = point_cam[1] + offsetX
        point_cam[2] = point_cam[2] + offsetY

        # Override for simulator
        # based on discussion https://discussions.udacity.com/t/focal-length-wrong/358568/22
        if fx < 10:
            fx = 2574
            fy = 2744
            point_cam[2] -= 1.0
            cx = image_width/2 - 35
            cy = image_height + 50
        lx = -point_cam[1] * fx / point_cam[0];
        ly = -point_cam[2] * fy / point_cam[0];

        lx = int(lx + cx)
        ly = int(ly + cy)


        return (lx, ly)


    def get_light_state(self, light):
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        height, width, channels = cv_image.shape
        pos = light.pose.pose.position

        ''' # Decided to make classifier without cropping
        # Crop project traffic light position and crop image
        x_up, y_up = self.project_to_image_plane(pos, .5, 1)
        x_down, y_down = self.project_to_image_plane(pos, -.5, -1)
        # rospy.loginfo('x_up = {}, y_up = {}, x_down = {}, y_down = {}'.format(x_up, y_up, x_down, y_down))

        if x_up > width or y_down > height or x_up < 0 or y_down < 0:
            return TrafficLight.UNKNOWN

        cpy = cv_image.copy()

        if x_down is None or x_up is None or y_up is None or y_down is None:
            return TrafficLight.UNKNOWN
        if x_down - x_up < 20 or y_down-y_up < 40:
            return TrafficLight.UNKNOWN

        output_crop = cpy[int(y_up):int(y_down), int(x_up):int(x_down)]

        # Publish to debug topic
        self.publish_image_test(output_crop)

        # save cropped image
        crop_folder = os.path.join('.', 'output_images', self.record_name, 'crop')
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder)
        crop_img_fname = os.path.join(crop_folder, 'crop{}.jpg'.format(self.nr))
        cv2.imwrite(crop_img_fname, output_crop)
        self.nr = self.nr + 1
        '''

        # Get Classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self, image, pose):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_wp = -1
        light_positions = self.config['stop_line_positions']


        # Select the closest waypoint from lights array which was received from /vehicle/traffic_lights topic
        if (self.lights and self.waypoints):

            car_wp = helper.next_waypoint_idx(pose, self.waypoints.waypoints)

            lights_wp = [helper.closest_waypoint_idx(l.pose, self.waypoints.waypoints) for l in self.lights]
            lights_dists = [helper.wp_distance(car_wp, lwp, self.waypoints.waypoints) for lwp in lights_wp]
            closest_light = lights_dists.index(min(lights_dists))

            # rospy.loginfo('car_wp = {}'.format(car_wp))
            # rospy.loginfo('closest_light[{}] = {}, {}'.format(closest_light, self.lights[closest_light].state, lights_dists[closest_light]))


            # light = lights_wp[closest_light]
            light_wp = lights_wp[closest_light]
            light = self.lights[closest_light]

            rospy.loginfo('car_wp = {:5d}, light_wp = {:5d}, light_dist = {:>4.2f}'.format(car_wp, light_wp, lights_dists[closest_light]))

            # This we have only in simulator for testing
            state = self.lights[closest_light].state
            self.sim_state = state
            # rospy.loginfo('SIM: closest_light_wp = {}, state = {}'.format(light_wp, light.state))


        # For debug output only
        self.cv_image_test = self.bridge.imgmsg_to_cv2(image, "bgr8")

        if light:
            waypoints_num = len(self.waypoints.waypoints)
            light_dist = (light_wp - car_wp + waypoints_num) % waypoints_num
            light_dist_m = lights_dists[closest_light]

            cv2.putText(self.cv_image_test, 'Light Dist: {:.2f} m'.format(light_dist_m), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

            # Look at the image and classify light
            if self.light_detect_dist_min < light_dist_m < self.light_detect_dist_max:
            	state = self.get_light_state(light)
                # state = TrafficLight.RED ## DEBUG
                # rospy.loginfo('CLASSIFIER STATE: state = {}'.format(state))

                # Publish annotated image_test topic for debug
                cv2.putText(self.cv_image_test, 'State: {}'.format(state), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

                return light_wp, state



        # Publish just image to image_test
        # self.publish_image_test(cv_image)

        # self.waypoints = None # don't know why this line is here [Pavlo]
        return -1, TrafficLight.UNKNOWN

    # Publish image for debug output
    def publish_image_test(self, image):
        image_message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8") # , encoding="rgb8"
        image_message.header = Header()
        image_message.header.stamp = rospy.Time.now()
        image_message.header.frame_id = '/world'
        self.image_test.publish(image_message)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
