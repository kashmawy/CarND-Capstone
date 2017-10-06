#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import waypoint_lib.helper as helper

import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704 # mph to mps
# TARGET_SPEED = 8.0 * ONE_MPH
# MAX_DECEL = -5.0 # m/s/s
# MAX_ACC = 1.0 # m/s/s
# STOP_LINE_DIST = 20 # SIM

dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.curr_vel_cb, queue_size=1)


        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = None
        self.current_velocity = 0
        self.red_light_wp = -1

        self.target_speed = rospy.get_param('~target_speed') / 3.6 # mps

        self.stop_line_dist = rospy.get_param('~stop_line_dist', 6.) # default for site
        self.max_acc = rospy.get_param('~max_acc', 1.) # default for site
        self.max_decel = rospy.get_param('~max_dec', -1.) # default for site


        self.slowing_down = False

        # For Debugging
        self.cnt = 0

        rospy.spin()


    def curr_vel_cb(self, curr_vel_msg):
    #   rospy.loginfo("current_velocity = {}".format(curr_vel_msg.twist))
      self.current_velocity = curr_vel_msg.twist.linear.x


    def pose_cb(self, pose):
        # TODO: Implement
        # rospy.loginfo('pose cb!!!!')
        # rospy.loginfo('Current pose = {}'.format(pose))
        if self.waypoints is None:
            rospy.loginfo('None waypoints')
            return

        log_out = (self.cnt % 20 == 0)

        # dists = [self.dist_pose_waypoint(pose, wp) for wp in self.waypoints]
        # closest_waypoint = dists.index(min(dists))

        # Find the number of waypoints
        waypoints_num = len(self.waypoints)

        # Find the closer waypoint
        wp_next = helper.next_waypoint_idx(pose, self.waypoints)

        # final_waypoints = []
        final_waypoints = helper.clone_waypoints(self.waypoints, wp_next, LOOKAHEAD_WPS)

        target_speed = self.target_speed
        # target_speed = final_waypoints[0].twist.twist.linear.x

        # Find the distance to the next traffic light
        # Distance to red light in waypoints
        wps_to_light = (self.red_light_wp - wp_next + waypoints_num) % waypoints_num
        if self.red_light_wp > 0:
            dist_to_light = helper.wp_distance(wp_next, self.red_light_wp, self.waypoints)
            dist_to_stop_line = dist_to_light - self.stop_line_dist # m
        # if wps_to_light < 0:
        #     wps_to_light += waypoints_num

        # Place the next waypoint
        # Final waypoint of our path
        # la_wp = (wp_next + LOOKAHEAD_WPS) % waypoints_num

        # Site: if we have less waypoints than in the whole track
        # if LOOKAHEAD_WPS > waypoints_num:
            # la_wp = waypoints_num - 1

        # Distance to stop line
        # rl_stop_line_nearest = 20


        # Distance to red_light where to stop (in waypoints)
        # TODO: Make the whole stopping in front of a stop line smooth
        # need to adjust distances where to stop based on current velocity and
        # find out correct params for this. [Pavlo]
        # rl_stop_line = 55

        # Stop line waypoint (where speed = 0.0)
        # rl_stop_line_wp = (self.red_light_wp - rl_stop_line + waypoints_num) % waypoints_num


        uniform_speed = True
        if self.red_light_wp < 0:
            # There is no red light ahead, so just set up max speed
            if log_out: rospy.loginfo("no red light >>")
            uniform_speed = True
            target_speed = self.target_speed
            self.slowing_down = False
        elif wps_to_light > LOOKAHEAD_WPS:
            # Red light is farther than number of points to return,
            # so again use max speed for all waypoints
            if log_out: rospy.loginfo("no red light is further than lookahead >>")
            uniform_speed = True
            target_speed = self.target_speed
            self.slowing_down = False
        elif dist_to_stop_line + 2 < 0:
            # We've already passed stop line
            if log_out: rospy.loginfo("missed stop line ({}) >>".format(dist_to_stop_line))
            uniform_speed = True
            target_speed = self.target_speed
            self.slowing_down = False
        elif helper.calc_acc(self.current_velocity, 0.0, dist_to_stop_line + 2.0) < self.max_decel and not self.slowing_down:
            # We are moving to fast to make a full stop, just continue
            if log_out: rospy.loginfo("too fast to stop. all_decc = {}, max_decel = {} >>".format(helper.calc_acc(self.current_velocity, 0.0, dist_to_stop_line + 2.0), self.max_decel))
            uniform_speed = True
            target_speed = self.target_speed
        else:
            # Red light is ahead, need to change speed gradually
            if log_out: rospy.loginfo("red light ahead {:.2f} m, need to stop.".format(dist_to_light))
            uniform_speed = False
            self.slowing_down = True

        # if log_out:
        #     speed_list = ['{:.2f}'.format(w.twist.twist.linear.x) for w in final_waypoints]
        #     rospy.loginfo("final_waypoints_start[{}] = [{}]".format(len(final_waypoints), ", ".join(speed_list)))

        if uniform_speed:
            # Just move forward from current velocity to the desired one
            if log_out: rospy.loginfo("just move forward")
            max_desired_speed = self.target_speed # or 0
            helper.move_forward_waypoints(
                final_waypoints,
                self.current_velocity,
                max_desired_speed = max_desired_speed,
                max_acceleration = self.max_acc
            )

        else:
            # wp_next -- rl_stop_line_wp -- red_light_wp -- la_wp

            # 0 -- dist_to_stop_line --- dist_to_light --- la_wp

            if log_out: rospy.loginfo("decelerate and stop in {} m".format(dist_to_stop_line))
            helper.decelerate_waypoints(
                final_waypoints,
                self.current_velocity,
                stop_distance = dist_to_stop_line,
                max_deceleration = self.max_decel
            )


        # rospy.loginfo("one point = {}".format(self.waypoints[0]))

        # orientation = self.waypoints[closest_waypoint].pose.pose.orientation

        if log_out:
            # rospy.loginfo('final_waypoints[0] = {}'.format(final_waypoints[0]))
            rospy.loginfo("pose x, y, yaw = {}, {}, {}".format(pose.pose.position.x,
                pose.pose.position.y, helper.yaw_from_orientation(pose.pose.orientation)))
            # rospy.loginfo("next wp x, y   = {}, {}".format(final_waypoints[0].pose.pose.position.x,
            #     final_waypoints[0].pose.pose.position.y))
            # rospy.loginfo("next wp linear.x   = {}".format(final_waypoints[0].twist.twist.linear.x))
            rospy.loginfo('current_velocity = {:.4f}'.format(self.current_velocity))
            rospy.loginfo('wp_next = {}'.format(wp_next))
            rospy.loginfo('red_light_wp = {}'.format(self.red_light_wp))
            if self.red_light_wp > 0:
                rospy.loginfo("dist_to_stop_line = {} m".format(dist_to_stop_line))
                rospy.loginfo("dist_to_light = {} m".format(dist_to_light))
            rospy.loginfo("lookahead_dist = {} m".format(helper.wp_distance(0, len(final_waypoints)-1, final_waypoints)))
            # rospy.loginfo('dist to zero = {}'.format(wps_to_light - decel_len))
            # rospy.loginfo('len wp = {}'.format(len(final_waypoints)))

            # rospy.loginfo("dist min = [{}] = {}".format(closest_waypoint, dists[closest_waypoint]))
            # rospy.loginfo("yaw = {}".format(helper.yaw_from_orientation(orientation)))

            speed_list = ['{:.2f}'.format(w.twist.twist.linear.x) for w in final_waypoints]
            rospy.loginfo("final_waypoints[{}] = [{}]".format(len(final_waypoints), ", ".join(speed_list)))

        self.cnt += 1

        self.publish(final_waypoints)

    def publish(self, waypoints):
      lane = Lane()
      lane.header.frame_id = '/world'
      lane.header.stamp = rospy.Time(0)
      lane.waypoints = waypoints
      self.final_waypoints_pub.publish(lane)


    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.waypoints = waypoints.waypoints
        # rospy.loginfo('received waypoints len = {}'.format(len(waypoints.waypoints)))


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.red_light_wp = int(msg.data)
        rospy.loginfo('received red_light_wp = {}'.format(self.red_light_wp))


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
