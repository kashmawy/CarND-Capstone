import tf
from tf import transformations as t
import math
import rospy
import numpy as np
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane, Waypoint

dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

def yaw_from_orientation(o):
    # https://answers.ros.org/question/69754/quaternion-transformations-in-python/
    q = (o.x, o.y, o.z, o.w)
    return tf.transformations.euler_from_quaternion(q)[2]


def dist_pose_waypoint(pose, waypoint):
    return dl(pose.pose.position, waypoint.pose.pose.position)

def closest_waypoint_idx(pose, waypoints):
    dists = [dist_pose_waypoint(pose, wp) for wp in waypoints]
    closest_waypoint = dists.index(min(dists))
    return closest_waypoint

def next_waypoint_idx(pose, waypoints):
    # dists = [dist_pose_waypoint(pose, wp) for wp in waypoints]
    # closest_waypoint = dists.index(min(dists))
    closest_waypoint = closest_waypoint_idx(pose, waypoints)

    waypoints_num = len(waypoints)

    wp = waypoints[closest_waypoint]

    pose_orientation = pose.pose.orientation

    # wp_yaw = helper.yaw_from_orientation(wp_orientation)
    pose_yaw = yaw_from_orientation(pose_orientation)

    angle = math.atan2(wp.pose.pose.position.y-pose.pose.position.y, wp.pose.pose.position.x-pose.pose.position.x)
    # rospy.loginfo('angle1 = {}'.format(angle))
    # rospy.loginfo('pose_yaw = {}'.format(pose_yaw))
    delta = abs(pose_yaw-angle)
    while delta > math.pi: delta -= math.pi
    # rospy.loginfo("delta1 = {}".format(delta))
    if (delta > math.pi/4):
        closest_waypoint = (closest_waypoint + 1) % waypoints_num
        wp = waypoints[closest_waypoint]
        # rospy.loginfo('forward')

    # angle = math.atan2(wp.pose.pose.position.y-pose.pose.position.y, wp.pose.pose.position.x-pose.pose.position.x)
    # delta = abs(pose_yaw-angle)
    # while delta > math.pi: delta -= math.pi

    # rospy.loginfo('angle = {}'.format(angle))
    # rospy.loginfo("delta = {}".format(delta))
    # rospy.loginfo('wp_yaw = {}'.format(wp_yaw))
    return closest_waypoint


def calc_steer_cte(pose, waypoints, fit_length = 10):

    if not fit_length:
        fit_length = len(waypoints)

    if fit_length > len(waypoints):
        return 0.0

    # Get X,Y coords
    x_coords = []
    y_coords = []
    for i in range(fit_length):
        x_coords.append(waypoints[i].pose.pose.position.x)
        y_coords.append(waypoints[i].pose.pose.position.y)

    # Transform to car coordinates
    x_coords_car, y_coords_car = tranform_to_pose_coord_xy(pose, x_coords, y_coords)

    coeffs = np.polyfit(x_coords_car, y_coords_car, 3)
    dist = np.polyval(coeffs, 0.0)

    return dist


def tranform_to_pose_coord_xy(pose, x_coords, y_coords):
    x_coords_pose = []
    y_coords_pose = []
    pose_x = pose.pose.position.x
    pose_y = pose.pose.position.y
    pose_yaw = yaw_from_orientation(pose.pose.orientation)
    for x, y in zip(x_coords, y_coords):
        # Translation
        rx = x - pose_x
        ry = y - pose_y
        # Rotation
        rxf = rx * math.cos(pose_yaw) + ry * math.sin(pose_yaw)
        ryf = rx * (-1.0*math.sin(pose_yaw)) + ry * math.cos(pose_yaw)
        x_coords_pose.append(rxf)
        y_coords_pose.append(ryf)
    return x_coords_pose, y_coords_pose

# moving from wp1 to wp2
def wp_distance(wp1, wp2, waypoints):
    waypoints_num = len(waypoints)
    dist = 0
    dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

    curr_wp = wp1
    while curr_wp != wp2:
        next_wp = (curr_wp + 1) % waypoints_num
        dist += dl(waypoints[curr_wp].pose.pose.position, waypoints[next_wp].pose.pose.position)
        curr_wp = next_wp

    # for i in range(wp1, wp2+1):
    #     dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
    #     wp1 = i

    return dist


def get_inverse_trans_rot(pose):
    # Car transform
    transT_car = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
    rotT_car = (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w)

    # World transform
    transform = t.concatenate_matrices(t.translation_matrix(transT_car), t.quaternion_matrix(rotT_car))
    inversed_transform = t.inverse_matrix(transform)
    translation = t.translation_from_matrix(inversed_transform)
    quaternion = t.quaternion_from_matrix(inversed_transform)
    # rospy.loginfo('transT_world = {}'.format(translation))
    # rospy.loginfo('rotT_world = {}'.format(quaternion))
    # transT = translation
    # rotT = quaternion
    return translation, quaternion


def clone_waypoint(waypoint):
  w = Waypoint()
  w.pose.pose.position.x = waypoint.pose.pose.position.x
  w.pose.pose.position.y = waypoint.pose.pose.position.y
  w.pose.pose.position.z = waypoint.pose.pose.position.z
  w.pose.pose.orientation.x = waypoint.pose.pose.orientation.x
  w.pose.pose.orientation.y = waypoint.pose.pose.orientation.y
  w.pose.pose.orientation.z = waypoint.pose.pose.orientation.z
  w.pose.pose.orientation.w = waypoint.pose.pose.orientation.w
  w.twist.twist.linear.x = waypoint.twist.twist.linear.x
  w.twist.twist.linear.y = waypoint.twist.twist.linear.y
  w.twist.twist.linear.z = waypoint.twist.twist.linear.z
  w.twist.twist.angular.x = waypoint.twist.twist.angular.x
  w.twist.twist.angular.y = waypoint.twist.twist.angular.y
  w.twist.twist.angular.z = waypoint.twist.twist.angular.z
  return w


def clone_waypoints(waypoints, start = 0, num = None):
  wlen = len(waypoints)
  if num is None or wlen < num:
    num = wlen
  new_waypoints = []
  idx = start
  for i in range(num):
    wid = (start + i) % wlen
    new_waypoints.append(clone_waypoint(waypoints[wid]))
  return new_waypoints

def calc_acc(v1, v2, dist):
  # if dist < 1e-06:
  #   return 0
  return (v2*v2 - v1*v1) / (2 * dist)

def move_forward_waypoints(
    final_waypoints,
    current_velocity,
    final_desired_speed = 0.0,
    max_acceleration = 1.0):

  d = 1.0 * dl(final_waypoints[0].pose.pose.position, final_waypoints[1].pose.pose.position)

  final_waypoints[0].twist.twist.linear.x = math.sqrt(2 * max_acceleration * d + current_velocity * current_velocity) # current_velocity
  for i in range(len(final_waypoints) - 1):

    w_prev = final_waypoints[i]
    v_prev = w_prev.twist.twist.linear.x
    w = final_waypoints[i+1]

    if final_desired_speed > 0.0:
      max_speed_cap = final_desired_speed
    else:
      max_speed_cap = w.twist.twist.linear.x

    dist = dl(w_prev.pose.pose.position, w.pose.pose.position)

    max_v = math.sqrt(2 * max_acceleration * dist + v_prev * v_prev)

    w.twist.twist.linear.x = min(max(max_v, 0), max_speed_cap)

  # TODO: Check return value


def decelerate_waypoints(
    final_waypoints,
    current_velocity,
    stop_distance = None,
    max_deceleration = -1.0):


  if stop_distance is None:
    stop_distance = wp_distance(0, len(final_waypoints) - 1, final_waypoints)

  all_dec = calc_acc(current_velocity, 0.0, stop_distance)

  if all_dec < max_deceleration:
    all_dec = max_deceleration

  final_waypoints[0].twist.twist.linear.x = current_velocity

  for i in range(len(final_waypoints) - 1):
    w_prev = final_waypoints[i]
    v_prev = w_prev.twist.twist.linear.x
    w = final_waypoints[i+1]

    dist = dl(w_prev.pose.pose.position, w.pose.pose.position)

    v_all_dec = 2 * all_dec * dist + v_prev * v_prev
    if v_all_dec > 0.0:
      v_all_dec = math.sqrt(v_all_dec)
    else:
      v_all_dec = 0.0

    w.twist.twist.linear.x = v_all_dec
