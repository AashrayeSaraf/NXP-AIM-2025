# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading
import queue

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = False

# =============================== Utility Functions =====================================================
def extract_occupied_points(grid):
	
	points =  np.column_stack(np.where(grid == 1))  # (y, x)
	points = points[:, ::-1] #(x, y)
	return points

def occupancy_grid_to_numpy(grid, threshold=0.1):
	width = grid.shape[1]
	height = grid.shape[0]
	data = np.array(grid.data).reshape((height, width))
	
	binary = np.where(data>threshold, 1, 0).astype(np.uint8)
	return binary

def cluster_points(points, eps=5, min_samples=10):
	db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
	labels = db.labels_
	clusters = []
	for label in set(labels):
		if label == -1:
			continue
		clusters.append(points[labels == label])
	return clusters

def merge_clusters(clusters, merge_eps=15):
	"""Merge centroids of nearby clusters using DBSCAN."""
	if len(clusters) <= 1:
		return clusters

	centroids = np.array([np.mean(c, axis=0) for c in clusters])
	db = DBSCAN(eps=merge_eps, min_samples=1).fit(centroids)
	merged = []
	for label in set(db.labels_):
		grouped = [clusters[i] for i in np.where(db.labels_ == label)[0]]
		merged.append(np.vstack(grouped))
	return merged

def is_rectangular(cluster, ratio_threshold=(1.4, 8), fill_threshold=0.3):
	"""Check if a cluster is likely rectangular based on PCA and area fill ratio."""
	if len(cluster) < 4:
		return False

	pca = PCA(n_components=2)
	aligned = pca.fit_transform(cluster)
	x_min, y_min = aligned.min(axis=0)
	x_max, y_max = aligned.max(axis=0)
	width, height = x_max - x_min, y_max - y_min

	if height == 0 or width == 0:
		return False

	aspect_ratio = max(width, height) / min(width, height)
	if not (ratio_threshold[0] <= aspect_ratio <= ratio_threshold[1]):
		return False

	bounding_area = width * height
	point_area = len(cluster)
	fill_ratio = point_area / bounding_area
	return fill_ratio >= fill_threshold

def get_oriented_bounding_box(cluster):
	"""Compute the rotated bounding box using PCA."""
	pca = PCA(n_components=2)
	transformed = pca.fit_transform(cluster)
	x_min, y_min = transformed.min(axis=0)
	x_max, y_max = transformed.max(axis=0)

	box = np.array([
		[x_min, y_min],
		[x_max, y_min],
		[x_max, y_max],
		[x_min, y_max]
	])
	return pca.inverse_transform(box).astype(int)

def get_min_area_rotated_box(cluster):
	"""
	Compute the minimum-area rotated bounding box aligned with object length.
	Returns 4 corner points.
	"""
	cluster = np.array(cluster, dtype=np.float32)  # Ensure float32 for cv2
	rect = cv2.minAreaRect(cluster)                # Get rotated rect: ((cx,cy), (w,h), angle)
	box = cv2.boxPoints(rect)                      # Get corner points
	box = np.intp(box)                             # Convert to integer
	return box

def draw_bounding_boxes(grid, boxes):
	"""Draw boxes on the original occupancy grid."""
	img_color = cv2.cvtColor((grid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
	for box in boxes:
		cv2.polylines(img_color, [box], isClosed=True, color=(0, 255, 0), thickness=2)
	return img_color

## ------- Identify the approach points ------------------
def unit_perpendicular(vec):
	"""Return a unit vector perpendicular to vec (90° clockwise)."""
	perp = np.array([vec[1], -vec[0]])
	return perp / np.linalg.norm(perp)

def unit_perpendicular(vec):
	"""Return a unit vector perpendicular to vec (90° clockwise)."""
	perp = np.array([vec[1], -vec[0]])
	return perp / np.linalg.norm(perp)

def order_rectangle_points(pts):
	"""Order 4 points of a quadrilateral in clockwise order around centroid."""
	center = np.mean(pts, axis=0)
	angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
	return pts[np.argsort(angles)]

def get_offset_points(rect, d):
	"""Return 1 offset point from length side and 1 from breadth side."""
	
	# Ensure the points are in order
	rect = order_rectangle_points(rect)

	# Extract sorted points
	p0, p1, p2, p3 = rect  # Assumed order: TL, TR, BR, BL (or similar consistent order)

	# Two adjacent edges
	edge1 = p1 - p0
	edge2 = p2 - p1

	# Determine which is length and which is width
	if np.linalg.norm(edge1) > np.linalg.norm(edge2):
		length_edge = (p0, p1)
		width_edge = (p1, p2)
	else:
		length_edge = (p1, p2)
		width_edge = (p2, p3)

	# Midpoints
	len_mid = (length_edge[0] + length_edge[1]) / 2
	wid_mid = (width_edge[0] + width_edge[1]) / 2

	# Perpendicular directions
	len_perp = unit_perpendicular(length_edge[1] - length_edge[0])
	wid_perp = unit_perpendicular(width_edge[1] - width_edge[0])

	# Offset points
	length_point = len_mid + len_perp * d
	width_point = wid_mid + wid_perp * d

	return length_point, width_point

def angle_with_x_axis(p1, p2):
	"""
	Returns the angle in radians between the line from p1 to p2 and the positive X-axis,
	measured in the anti-clockwise direction.

	Parameters:
		p1, p2: Tuples or arrays of the form (x, y)

	Returns:
		angle in radians in range [0, 2π)
	"""
	x1, y1 = p1
	x2, y2 = p2
	angle = np.arctan2(y2-y1, x2-x1)  # returns value in [-π, π]
	angle_deg = np.degrees(angle) % 360
	return angle_deg

def unit_vector_from_points(p1, p2):
	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]
	norm = math.hypot(dx, dy)
	if norm == 0:
		raise ValueError("Points must be distinct!")
	return (dx / norm, dy / norm)

def point_from_vector(origin, p1, p2, distance):
	"""
	Find a point in the direction of a vector at a given distance.

	Args:
		origin (tuple): Starting point (x0, y0) or (x0, y0, z0).
		direction (tuple): Direction vector (vx, vy) or (vx, vy, vz).
		distance (float): Distance to move along the direction.

	Returns:
		tuple: The new point at the specified distance.
	"""
	direction = unit_vector_from_points(p1, p2)
	length = math.sqrt(sum(c**2 for c in direction))
	if length == 0:
		raise ValueError("Direction vector cannot be zero.")
	unit_vector = tuple(c / length for c in direction)
	return tuple(o + distance * u for o, u in zip(origin, unit_vector))


class ObjectWithBBox:
	def __init__(self, _id, corners):
		self.id = _id
		assert len(corners)==4, "Bounding box must have exactly four points"
		self.corners = corners
		self.centroid = np.mean(corners, axis=0)
	
	def get_points_to_visit(self, d=30):
		"""
		This function computes and return two points to visit, one point will be in front of the QR code and the other in front of the objects.
		
		Returns: List[p1, p2]
		"""
		p1, p2 = get_offset_points(self.corners, d=d)
		return p1, p2

#==============================================================================================================

class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
						  relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)
		
		# Subscribe to shelf data (combined objects + QR codes)
		self.subscription_shelf_data = self.create_subscription(
			WarehouseShelf,
			'/shelf_data',
			self.shelf_data_callback,
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = self.get_parameter('initial_angle').get_parameter_value().double_value
		self.shelf_angle = self.initial_angle

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_failed = False # Was the last goal a success or a failure.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code and Shelf Data  ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0
			self.accumulated_objects = {}  # Store objects for each column: {column_id: [objects]}
			self.qr_filled = [False] * self.shelf_count      # Track if QR is filled in each column
			self.qr_detection_count = 0  # Track how many QRs have been detected
			self.detected_qr_codes = set()  # Track unique QR codes to avoid duplicates

		self.shelf_objects_curr = WarehouseShelf()
		self.qr_code_str = None
		self.shelf_objects_name = None
		self.shelf_objects_count = None

		# ---- Global Flags ----
		self.anchor = (0, 0) #initial position of the buggy
		self.shelf_angle = self.initial_angle
		self.identified_objects = None
		self.visited = set()
		self.visit_in_process = False
		#Are we in the middle of visiting a shelf
		self.scan_complete = True
		# A flag that tells the explorer function that scan in complete and now move to the other side.
		self.current_obj = None

		self.goal_progress= None
		self.exploration_state = 'EXPLORE_FRONTIER'
		self.scan_state = 'SCAN_MID_YAW'

		self.scan_duration = 0.5  # 1 seconds minimum scan time
		self.scan_start_time = 0.0
		self.scan_result = []
		self.qr_published = [False] * self.shelf_count
		self.shelfs_published = {i: False for i in range(self.shelf_count)}  # Track published shelves
		self.shelf_data = []

		self.target_shelf = 1 # Shelf that we are in search of.
		self.find_next_shelf = False # Is the buggy in search of next shelf 
		self.next_shelf_candidates = queue.Queue() # queue of ObjectWithBBox that are potentially next shelf
		self.decision_state = 'FIND_NEXT_SHELF_CANDIDATES'
		# FIND_NEXT_SHELF_CANDIDATES, MOVE, WAIT, SCAN, DONE
		self.candidate_obj = None
		self.all_shelfs_explored = False
		self.shelf_angle_tolerance = 10

	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)
		# self.logger.info(f"Buggy position updated: x={self.buggy_pose_x}, y={self.buggy_pose_y}")

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(map_info.width / 2, map_info.height / 2, map_info)

	def explore_frontier(self):
		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		frontiers = self.get_frontiers_for_space_exploration(map_array)

		map_info = self.global_map_curr.info
		if frontiers:
			closest_frontier = None
			min_distance_curr = float('inf')

			for fy, fx in frontiers:
				fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy,
											 map_info)
				distance = euclidean((fx_world, fy_world), self.buggy_center)
				if (distance < min_distance_curr and
					distance <= self.max_step_dist_world_meters and
					distance >= self.min_step_dist_world_meters):
					min_distance_curr = distance
					closest_frontier = (fy, fx)

			if closest_frontier:
				fy, fx = closest_frontier
				goal = self.create_goal_from_map_coord(fx, fy, map_info)
				self.send_goal_from_world_pose(goal)
				print("Sending goal for space exploration.")
				return
			else:
				self.max_step_dist_world_meters += 2.0
				new_min_step_dist = self.min_step_dist_world_meters - 1.0
				self.min_step_dist_world_meters = max(0.25, new_min_step_dist)

			self.full_map_explored_count = 0
		else:
			self.full_map_explored_count += 1
			print(f"Nothing found in frontiers; count = {self.full_map_explored_count}")
		
	def identify_rectangles(self, map_array): 
		map_array = occupancy_grid_to_numpy(map_array)
		occupied = extract_occupied_points(map_array)
	
		# Step 2: Initial clustering
		clusters = cluster_points(occupied, eps=4, min_samples=10)

		# Step 3: Merge close clusters
		merged_clusters = merge_clusters(clusters, merge_eps=20)
		self.logger.info(f"Number of clusters identified: {len(merged_clusters)}")

		# Step 4: Filter rectangular clusters
		rectangular_clusters = [c for c in merged_clusters if is_rectangular(c)]
		self.logger.info(f"Number of rectangular clusters identified: {len(rectangular_clusters)}")

		# Step 5: Get bounding boxes
		boxes = [get_oriented_bounding_box(c) for c in rectangular_clusters]

		return boxes

	def identify_objects_from_map(self):
		"""
		This function processes the explored global costmap and identify objects that are approximate rectangles (shelfs are 
		rectangles) and return a list of ObjectWithBBox objects each representing an identified object.

		"""
		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))
		rectangles = self.identify_rectangles(map_array)
		self.logger.info(f"LENGTH OF RECTANGLES IDENTIFIED = {len(rectangles)}")
		identified_objects = [] # List of ObjectWithBBox
		for i, rect in enumerate(rectangles):
			obj = ObjectWithBBox(_id=i, corners=rect)
			self.logger.info(f"[SHELF IDENTIFICATION] Shelfs Identified at {obj.centroid} ")
			identified_objects.append(obj)
			
		self.identified_objects = identified_objects
		# Visualize all the identified objects with their bounding boxes.
		gray_image = (map_array * 255).astype(np.uint8)
		image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
		
		# Plot bounding boxes and points to visit
		for obj in self.identified_objects:
			center_x, center_y = int(obj.centroid[0]), int(obj.centroid[1])
			cv2.circle(image, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)  # green filled circle
			ordered = order_rectangle_points(np.array(obj.corners, dtype=np.float32)).astype(int)
			cv2.polylines(image, [ordered], isClosed=True, color=(0, 255, 0), thickness=2)
			
			p1, p2 = obj.get_points_to_visit()
			p1_x, p1_y = int(p1[0]), int(p1[1])
			p2_x, p2_y = int(p2[0]), int(p2[1])	
			cv2.circle(image, (p1_x, p1_y), radius=5, color=(0, 255, 0), thickness=-1) 
			cv2.circle(image, (p2_x, p2_y), radius=5, color=(0, 255, 0), thickness=-1)

		cv2.imwrite("identified_objects.png", image)
		return
	
	def explore_object(self, obj):
		"""
		This function is responsible for exploring the object received by its parameter
		"""
		# Store exploration state as class variables
		self.exploration_state = 'MOVE_TO_FIRST_POINT'
		self.current_obj = obj
		
		# Start the state machine
		self.exploration_timer = self.create_timer(0.5, self.exploration_state_machine)
	
	def handle_scan_single_yaw(self, next_scan_state:str):
		current_time = time.time()
		scan_elapsed = current_time - self.scan_start_time
		
		# Check if minimum scan time has elapsed
		if scan_elapsed >= self.scan_duration:
			if self.scan_complete:
				self.logger.info(f"[SCAN] Scan complete after {scan_elapsed:.2f}s")
				if self.exploration_state == 'SCANNING_QR' and self.qr_code_str is not None and self.qr_code_str != "Empty":
					self.scan_state = 'SCAN_DONE'
					self.scan_complete = False
				else:
					self.scan_state = next_scan_state
					self.scan_complete = False
				# self.scan_start_time = time.time()  # Reset scan start time for next scan
			else:
				# self.scan_start_time = time.time()  # Reset scan start time for next scan
				self.scan_state = next_scan_state
		else:
			remaining_time = self.scan_duration - scan_elapsed
			self.logger.info(f"[SCAN] Scanning in progress, {remaining_time:.2f}s remaining")
	
	def exploration_state_machine(self):
		"""
		State machine to handle object exploration without blocking callbacks
		"""
		if not hasattr(self, 'exploration_state'):
			self.exploration_timer.cancel()
			return

		map_info = self.global_map_curr.info
		p1, p2 = self.current_obj.get_points_to_visit()

		if self.exploration_state == 'MOVE_TO_FIRST_POINT':
			self.logger.info(f"[SHELF EXPLORATION] Points to visit -> {p1}, {p2}")
			
			fx, fy = int(p1[0]), int(p1[1])
			yaw = angle_with_x_axis([fx, fy], self.current_obj.centroid) * np.pi / 180.0
	
			self.logger.info(f"[SHELF EXPLORATION] Visiting the first point {p1} at orientation {yaw}")
			goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw)
			self.send_goal_from_world_pose(goal)
			self.exploration_state = 'WAITING_FOR_FIRST_POINT'
			
		elif self.exploration_state == 'WAITING_FOR_FIRST_POINT':
			if self.goal_failed:
				self.logger.info(f"[SHELF EXPLORATION] Goal failed. Resending the same goal")
				fx, fy = int(p1[0]), int(p1[1])
				yaw = angle_with_x_axis([fx, fy], self.current_obj.centroid) * np.pi / 180.0		
				goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw)
				self.send_goal_from_world_pose(goal)
				self.exploration_state = 'WAITING_FOR_FIRST_POINT'

			elif self.goal_completed:
				self.logger.info(f" [SHELF EXPLORATION] Reached first point, initiating scan  ========")
				# self.logger.info(f"\t Buggy pose x: {self.buggy_pose_x}, y: {self.buggy_pose_y}, orientation: {self.pose_curr.pose.pose.orientation}")
				# self.logger.info(f"\t [POSE] Buggy orientation: {self.quaternion_to_yaw(self.pose_curr.pose.pose.orientation)}") 
				self.scan_complete = False
				self.exploration_state = 'SCANNING_SHELF_OBJECTS'
				self.scan_start_time = time.time()
				
		elif self.exploration_state == 'SCANNING_SHELF_OBJECTS':
			fx, fy = int(p1[0]), int(p1[1])
			yaw = angle_with_x_axis([fx, fy], self.current_obj.centroid) * np.pi / 180.0
			if self.scan_state == 'SCAN_MID_YAW':
				self.logger.info("[SCAN] Scanning mid yaw ")
				self.handle_scan_single_yaw('ROTATE_LEFT_YAW')
			elif self.scan_state == 'ROTATE_LEFT_YAW':
				fx, fy = point_from_vector([fx, fy], self.current_obj.centroid, p1, 10)
				self.logger.info(" [SCAN] Sending goal to rotate left ")
				goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw + (20*np.pi/180.0))
				self.send_goal_from_world_pose(goal)
				self.scan_state = 'WAIT_FOR_LEFT_YAW'
					
			elif self.scan_state == 'WAIT_FOR_LEFT_YAW':
				self.logger.info("[SCAN] Waiting for left yaw rotation to complete ")
				if self.goal_completed:
					self.scan_state = 'SCAN_LEFT_YAW'
					self.scan_start_time = time.time()  # Reset scan start time for next scan
			elif self.scan_state == 'SCAN_LEFT_YAW':
				self.logger.info("[SCAN] Scanning left yaw ")
				self.handle_scan_single_yaw('ROTATE_RIGHT_YAW')
			elif self.scan_state == 'ROTATE_RIGHT_YAW':
				fx, fy = point_from_vector([fx, fy], self.current_obj.centroid, p1, 20)
				self.logger.info("[SCAN] Sending goal to rotate right ")
				goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw - (20*np.pi/180.0))
				self.send_goal_from_world_pose(goal)
				self.scan_state = 'WAIT_FOR_RIGHT_YAW'
			elif self.scan_state == 'WAIT_FOR_RIGHT_YAW':
				self.logger.info("[SCAN] Waiting for right yaw rotation to complete")
				if self.goal_completed:
					self.scan_state = 'SCAN_RIGHT_YAW'
					self.scan_start_time = time.time()  # Reset scan start time for next scan
			elif self.scan_state == 'SCAN_RIGHT_YAW':
				self.logger.info(" [SCAN] Scanning right yaw ========")
				self.handle_scan_single_yaw('SCAN_DONE')
				self.logger.info("[SCAN] Scanning right yaw complete, Moving to second point")

			elif self.scan_state == 'SCAN_DONE':
				# self.exploration_state = 'MOVE_TO_SECOND_POINT'
				self.exploration_state = 'DONE'
				self.scan_state = 'SCAN_MID_YAW'  # Reset scan state for next exploration
				self.logger.info(f" [SCAN] OBJECT SCAN COMPLETE")
				# self.reconcile_shelf_objects()
				self.scan_result = []
				if self.shelf_objects_name is None or self.shelf_objects_count is None:
					self.logger.info("[SCAN] No objects found on the shelf")
				else:
					for obj_name, obj_count in zip(self.shelf_objects_name, self.shelf_objects_count):
						self.logger.info(f"\tObject Name: {obj_name}, Count: {obj_count}")
										
		elif self.exploration_state == 'DONE':
			# Publishing data to /shelf_data
			if self.qr_code_str is not None and self.qr_code_str != "Empty":
				shelf_id = int(self.qr_code_str.split("_")[0])
				if self.is_shelf_unlocked(self.qr_code_str) and shelf_id < self.shelf_count:
					# Publish the data if the shelf is not the last shelf.
					self.publish_shelf_data()
				elif shelf_id == self.shelf_count:
					self.logger.info(f"[SHELF DATA] Saving last shelf data for later publishing.")
					shelf = WarehouseShelf() 
					shelf.qr_decoded = self.qr_code_str 
					shelf.object_name = self.shelf_objects_name 
					shelf.object_count = self.shelf_objects_count 
					self.shelf_data.append(shelf)

				# If shelf is locked publishes atleast the QR
				else:		
					shelf_id = int(self.qr_code_str.split("_")[0])
					self.logger.info(f"[SHELF DATA] Shelf is locked. Will try to revisit it.")
					if not self.qr_published[shelf_id-1] and shelf_id < self.shelf_count:
						message = WarehouseShelf()
						message.qr_decoded = self.qr_code_str 
						self.publisher_shelf_data.publish(message)
						self.qr_published[shelf_id-1] = True
					
			# Clean up and finish exploration
			self.logger.info(f"[SHELF EXPLORATION] Object exploration complete")
			self.exploration_timer.cancel()
			# Notify that the exploration is complete (needed for explore_next_object)
			self.visit_in_process = False
			self.visited.add(self.current_obj.id)
			self.current_obj = None
			self.qr_code_str = None
			self.shelf_objects_name = None
			self.shelf_objects_count = None
			self.scan_state = 'SCAN_MID_YAW'  # Reset scan state for next exploration
			self.find_next_shelf = True 
			self.decision_state = 'FIND_NEXT_SHELF_CANDIDATES'
			self.target_shelf += 1

	def get_next_shelf_candidates(self):
		candidates = []
		self.logger.info(f"[CHOOSE NEXT SHELF] Determining candidates for {self.target_shelf}")
		for obj in self.identified_objects:	
			if obj.id not in self.visited:
				obj_centroid_world_coordinates = self.get_world_coord_from_map_coord(obj.centroid[0], obj.centroid[1], self.global_map_curr.info)
				theta = angle_with_x_axis(self.anchor, obj_centroid_world_coordinates)
				self.logger.info(f"\t Centroid: {obj.centroid}, Computed Angle: {theta}")
				if min(abs(self.shelf_angle - theta), abs(self.shelf_angle-(360-theta))) < self.shelf_angle_tolerance:
					candidates.append(obj)
		
		sorted_candidates = sorted(candidates, key=lambda x: euclidean(self.anchor, self.get_world_coord_from_map_coord(x.centroid[0], x.centroid[1], self.global_map_curr.info)))
		candidates_queue = queue.Queue()
		for cand in sorted_candidates:
			candidates_queue.put(cand)

		self.logger.info(f"[CHOOSE NEXT SHELF] Found {len(candidates)} candidates for next shelf exploration")	
		self.next_shelf_candidates = candidates_queue
		return candidates
	

	def explore_next_object(self):
		if self.current_obj is not None:
			object_to_explore = self.current_obj
			self.logger.info(f"[CHOOSE NEXT SHELF] Initiating object with id {object_to_explore.id} exploration")
			self.visit_in_process = True
			self.current_obj = object_to_explore
			self.explore_object(object_to_explore)
		else:
			self.logger.info("[CHOOSE NEXT SHELF] No object found for exploration")
			self.all_shelfs_explored = True
	
	def handle_scan_single_yaw_2(self, next_scan_state:str):
		current_time = time.time()
		scan_elapsed = current_time - self.scan_start_time
		
		# Check if minimum scan time has elapsed
		if scan_elapsed >= self.scan_duration:
			if self.scan_complete:
				self.logger.info(f"[SCAN] Scan complete after {scan_elapsed:.2f}s")
				if self.decision_state == 'SCAN' and self.qr_code_str is not None and self.qr_code_str != "Empty":
					self.scan_state = 'SCAN_DONE'
					self.scan_complete = False
				else:
					self.scan_state = next_scan_state
					self.scan_complete = False
			else:
				self.scan_state = next_scan_state
		else:
			remaining_time = self.scan_duration - scan_elapsed
			self.logger.info(f"[SCAN] Scanning in progress, {remaining_time:.2f}s remaining")

	def determine_next_shelf(self):
		
		map_info = self.global_map_curr.info
		if self.decision_state == 'FIND_NEXT_SHELF_CANDIDATES':
			self.get_next_shelf_candidates()
			self.decision_state = 'MOVE'
		elif self.decision_state == 'MOVE':	
			if self.next_shelf_candidates.empty():
				self.logger.info("[FIND NEXT SHELF] No more candidates found, stopping exploration.")
				self.find_next_shelf = False
				return

			obj = self.next_shelf_candidates.get()
			self.candidate_obj = obj
			p1, p2 = obj.get_points_to_visit()	
			fx, fy = int(p2[0]), int(p2[1])
			yaw = angle_with_x_axis([p2[0], p2[1]], obj.centroid) * np.pi/180.0
			self.logger.info(f"[FIND NEXT SHELF] Sending goal to visit the candidate")
			goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw)
			self.send_goal_from_world_pose(goal)
			self.decision_state = 'WAIT'
		elif self.decision_state == 'WAIT':
			if self.goal_completed:
				self.logger.info(f"[FIND NEXT SHELF] Candidate shelf reached")
				self.scan_complete = False
				self.scan_start_time = time.time()
				self.decision_state = 'SCAN'
				self.scan_state = 'SCAN_MID_YAW'

		elif self.decision_state == 'SCAN':
			p1, p2 = self.candidate_obj.get_points_to_visit(d=20)
			fx, fy = int(p2[0]), int(p2[1])
			yaw = angle_with_x_axis([p2[0], p2[1]], self.candidate_obj.centroid) * np.pi / 180.0
			if self.scan_state == 'SCAN_MID_YAW':
				self.logger.info("[SCAN] Scanning mid yaw")
				self.handle_scan_single_yaw_2('ROTATE_LEFT_YAW')
			elif self.scan_state == 'ROTATE_LEFT_YAW':
				fx, fy = point_from_vector([p2[0], p2[1]], self.candidate_obj.centroid, p2, 15)
				yaw = (angle_with_x_axis([p2[0], p2[1]], self.candidate_obj.centroid) + 20) * np.pi / 180.0
				self.logger.info(" [SCAN] Sending goal to rotate left ")
				goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw)
				self.send_goal_from_world_pose(goal)
				self.scan_state = 'WAIT_FOR_LEFT_YAW'
			elif self.scan_state == 'WAIT_FOR_LEFT_YAW':
				self.logger.info("[SCAN] Waiting for left yaw rotation to complete ")
				if self.goal_completed:
					self.scan_state = 'SCAN_LEFT_YAW'
					self.scan_start_time = time.time()  # Reset scan start time for next scan
			elif self.scan_state == 'SCAN_LEFT_YAW':
				self.logger.info("[SCAN] Scanning left yaw ")
				self.handle_scan_single_yaw_2('ROTATE_RIGHT_YAW')
			elif self.scan_state == 'ROTATE_RIGHT_YAW':
				fx, fy = point_from_vector([p2[0], p2[1]], self.candidate_obj.centroid, p2, 30)
				yaw = (angle_with_x_axis([p2[0], p2[1]], self.candidate_obj.centroid)- 20)*np.pi/180.0
				self.logger.info("[SCAN] Sending goal to rotate right ")
				goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw)
				self.send_goal_from_world_pose(goal)
				self.scan_state = 'WAIT_FOR_RIGHT_YAW'
			elif self.scan_state == 'WAIT_FOR_RIGHT_YAW':
				self.logger.info("[SCAN] Waiting for right yaw rotation to complete")
				if self.goal_completed:
					self.scan_state = 'SCAN_RIGHT_YAW'
					self.scan_start_time = time.time()  # Reset scan start time for next scan
			elif self.scan_state == 'SCAN_RIGHT_YAW':
				self.logger.info(" [SCAN] Scanning right yaw ========")
				self.handle_scan_single_yaw_2('SCAN_DONE')
				self.logger.info("[SCAN] Scanning right yaw complete, Moving to second point")

			elif self.scan_state == 'SCAN_DONE':
				self.scan_state = 'SCAN_MID_YAW'  # Reset scan state for next exploration
				self.logger.info(f" [SCAN] QR SCAN COMPLETE")
				self.decision_state = 'DONE'
				self.logger.info(f"[FIND NEXT SHELF] Candidate shelf scan complete, checking if it is the target shelf.")
				self.logger.info(f"[FIND NEXT SHELF] Candidate shelf QR Code: {self.qr_code_str}")

		elif self.decision_state == 'DONE':
			self.logger.info(f"[FIND NEXT SHELF] Candidate shelf exploration complete")
			if self.qr_code_str is not None:
				shelf_id = int(self.qr_code_str.split("_")[0])
				if shelf_id == self.target_shelf:
					self.logger.info(f"[FIND NEXT SHELF] Found the target shelf {self.target_shelf}")
					self.current_obj = self.candidate_obj 
					self.find_next_shelf = False
					self.decision_state=  'FIND_NEXT_SHELF_CANDIDATES'
				else:
					self.decision_state = 'MOVE'
					self.candidate_obj = None
			else:
				self.logger.info(f"[FIND NEXT SHELF] self.qr_code_str is None.")		
				self.find_next_shelf = False 
				self.decision_state = 'FIND_NEXT_SHELF_CANDIDATES'

	
	def compute_percent_explored(self):
		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))
		
		explored_cells_count = 0 
		for i in range(len(map_array)):
			for j in range(len(map_array[0])):
				if(map_array[i][j]==-1):
					continue
				else:
					explored_cells_count += 1
		ratio = explored_cells_count / (height*width)
		self.logger.info(f"[MAP EXPLORATION] {ratio*100:.2f}% complete")
		
	
	def global_map_callback(self, message):
		"""
		Callback function to handle global map updates.

		Args:
			message: ROS2 message containing the global map data.

		Returns:
			None
		"""

		self.global_map_curr = message

		if not self.goal_completed:
			return
	
		if self.full_map_explored_count == 0:
			self.compute_percent_explored()
			self.explore_frontier()
			return 
		elif self.identified_objects is None:
			height, width = self.simple_map_curr.info.height, self.simple_map_curr.info.width
			map_array = np.array(self.global_map_curr.data).reshape((height, width))
			with open("map_array_4.csv", 'w') as f:
				for row in map_array:
					for j, col in enumerate(row):
						f.write(f"{col}")
						if j < width-1:
							f.write(", ")
					f.write("\n")
							
						
			self.logger.info("[MAP EXPLORATION] Global map exploration complete, now identifying objects from the map.")
			self.identify_objects_from_map()
			self.logger.info("[SHELF IDENTIFICATION] Identified objects from the map.")
			self.logger.info(f"[SHELF IDENTIFICATION] LENGTH OF IDENTIFIED OBJECTS = {len(self.identified_objects)}")
			# self.logger.info(f"[SHELF IDENTIFICATION] Objects: {self.identified_objects}")
			self.find_next_shelf = True
			self.decision_state = 'FIND_NEXT_SHELF_CANDIDATES'
			return
		# determine the next shelf to visit
		elif self.find_next_shelf and self.target_shelf <= self.shelf_count:
			self.determine_next_shelf()
		elif self.visit_in_process:
			return
		elif not self.all_shelfs_explored and self.target_shelf <= self.shelf_count:
			self.explore_next_object()
		elif not self.shelfs_published[self.shelf_count-1]:
			# If last shelf is not published
			self.logger.info("[SHELF DATA] Last shelf data is not published yet, publishing now.")
			for shelf in self.shelf_data:
				self.logger.info(f"[SHELF DATA] Shelf QR Code: {shelf.qr_decoded}")
				shelf_id = int(shelf.qr_decoded.split("_")[0])
				if shelf_id != self.shelf_count:
					continue
				if shelf.qr_decoded == '':
					self.logger.info("[SHELF DATA] Last shelf data is incomplete, cannot publish.")
					return
				shelf_id = int(shelf.qr_decoded.split("_")[0])
				self.logger.info(f"[SHELF DATA] Publishing the final shelf.")
				self.publisher_shelf_data.publish(shelf)
				self.shelfs_published[shelf_id-1] = True
		else:
			self.logger.info("[SHELF EXPLORATION] All shelves have been explored and published.")
			   
	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers

	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""Callback function to handle incoming camera images.

		Args:
			message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

		Returns:
			None
		"""
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		# Process the image from front camera as needed.

		# Optional line for visualizing image on foxglove.
		# self.publish_debug_image(self.publisher_qr_decode, image)

	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def is_shelf_unlocked(self, shelf_qr:str):
		shelf_id = int(shelf_qr.split("_")[0])
		for i in range(shelf_id-1):
			if not self.qr_published[i]:
				return False 
		return True
	
	def publish_shelf_data(self):
		"""
		This function will attempt to publish shelf data to /shelf_data topic

		Returns:
			True, if successfully publishes
			False, if information is incomplete 
		"""
		if self.qr_code_str is None or self.shelf_objects_name is None or self.shelf_objects_count is None:
			self.logger.info("[SHELF PUBLISHER] Incomplete shelf data, cannot publish.")
			if self.qr_code_str is None:
				self.logger.info("\t --- QR is Missing ---")
			if self.shelf_objects_name is None:
				self.logger.info("\t --- Shelf Objects Name is Missing ---")
			if self.shelf_objects_count is None:
				self.logger.info("\t --- Shelf Objects Count is Missing ---") 
			return False

		shelf_data_message = WarehouseShelf()

		shelf_data_message.object_name = self.shelf_objects_name
		shelf_data_message.object_count = self.shelf_objects_count
		shelf_data_message.qr_decoded = self.qr_code_str

		# Updating global flags
		self.shelf_angle = float(self.qr_code_str.split("_")[1]) #QR is of the form 1_259.7_Ad....
		self.anchor = self.get_world_coord_from_map_coord(self.current_obj.centroid[0], self.current_obj.centroid[1], self.global_map_curr.info)

		self.logger.info(f"[SHELF PUBLISHER] Publishing following data to /shelf_data")
		self.logger.info(f"\t QR - {self.qr_code_str}")
		self.logger.info(f"\t Objects Name - {self.shelf_objects_name}")
		self.logger.info(f"\t Objects Count - {self.shelf_objects_count}")
		self.publisher_shelf_data.publish(shelf_data_message)
		
		shelf_id = int(self.qr_code_str.split("_")[0])  # Extract shelf ID from QR code string
		self.qr_published[int(shelf_id)-1] = True  # Mark QR as published
		self.shelfs_published[int(shelf_id)-1] = True
		self.shelf_data.append(shelf_data_message)  # Store shelf data in the list
		
		return True
	
	def reconcile_shelf_objects_probabilistic(self):
		"""
		Merges the current shelf object data with new detected data, updating counts by summing them.
		"""
		scanned_objects = {}
		for name, count in self.scan_result:
			scanned_objects[name] = max(scanned_objects.get(name, 0), count)

		unique_shelf_objects = {}
		for name in scanned_objects.keys():
			count = scanned_objects[name]
			if name=='bowl':
				name = 'cup'
			elif name == 'vase':	
				name = 'potted plant'

			sca = f"{name}_{count}"
			if name not in unique_shelf_objects.keys():
				unique_shelf_objects[sca] = 1
			else:
				unique_shelf_objects[sca] += 1	
		
		scanned_objects_with_prob = []
		for scanned_object in unique_shelf_objects.keys():
			freq = unique_shelf_objects[scanned_object]
			prob = freq / len(scanned_objects.keys())
			scanned_objects_with_prob.append((scanned_object, prob))
		
		sorted_objects = sorted(scanned_objects_with_prob, key=lambda x: x[1], reverse=True)
		
		shelf_objects_name = []
		shelf_objects_count = []
		rem = 6
		for scanned_object, prob in sorted_objects[:min(6, len(sorted_objects))]:
			name, count = scanned_object.split("_")
			count = int(count)
			rem -= count
			shelf_objects_name.append(name)
			shelf_objects_count.append(count)
			if rem <= 0:
				break

		self.shelf_objects_name = shelf_objects_name
		self.shelf_objects_count = shelf_objects_count		
		self.logger.info(f"[SCAN RESULT] Shelf objects reconciled")

	def reconcile_shelf_objects(self, shelf_objects_name, shelf_objects_count):
		"""
		Merges the current shelf object data with new detected data, updating counts by summing them.
		
		Args:
			shelf_objects_name (List[str]): New objects detected.
			shelf_objects_count (List[int]): Corresponding object counts.
		"""
		if self.shelf_objects_name is None or self.shelf_objects_count is None:
			self.shelf_objects_name = shelf_objects_name.copy()
			self.shelf_objects_count = shelf_objects_count.copy()
			return

		# Create a dictionary of current objects
		current_objects = {
			name: count for name, count in zip(self.shelf_objects_name, self.shelf_objects_count)
		}

		# Reconcile new objects
		for name, count in zip(shelf_objects_name, shelf_objects_count):
			if name in current_objects:
				current_objects[name] = max(current_objects[name], count)
			else:
				current_objects[name] = count

		# Update lists
		self.shelf_objects_name = list(current_objects.keys())
		self.shelf_objects_count = list(current_objects.values())



	def shelf_objects_callback(self, message):
		"""Callback function to handle shelf objects updates.

		Args:
			message: ROS2 message containing shelf objects data.

		Returns:
			None
		"""
		self.shelf_objects_curr = message

		shelf = self.shelf_objects_curr
		#check if we have received a QR code
		try:
			name, count = shelf.object_name[0], shelf.object_count[0]
			if(name[:2]=="QR" ):
				qr_code_str = name[3:]

				shelf_qr_message = WarehouseShelf()
				shelf_qr_message.qr_decoded = qr_code_str
				shelf_id = int(qr_code_str.split("_")[0])
				if(not self.qr_published[shelf_id-1] and shelf_id < self.shelf_count):
					self.publisher_shelf_data.publish(shelf_qr_message)
					self.qr_published[shelf_id-1] = True
		except:
			pass
		
		if self.exploration_state == 'SCANNING_SHELF_OBJECTS':
			if self.scan_state == 'SCAN_MID_YAW' or self.scan_state == 'SCAN_LEFT_YAW' or self.scan_state == 'SCAN_RIGHT_YAW':
				try:
					name, count = shelf.object_name[0], shelf.object_count[0]
				except:
					self.logger.info("[SCAN RESULT] No objects detected in the current scan.")
					return
				
				shelf_objects_name = []
				shelf_objects_count = []
		
				for name, count in zip(shelf.object_name, shelf.object_count):
					if name == 'bowl':
						continue  # Skip bowl object
					# self.scan_result.append((name, count))
					shelf_objects_name.append(name)
					shelf_objects_count.append(count)
					self.logger.info(f"Object found: {name}: {count}")
					
				self.reconcile_shelf_objects(shelf_objects_name, shelf_objects_count)
				# self.shelf_objects_name = shelf_objects_name 
				# self.shelf_objects_count = shelf_objects_count
				self.scan_complete = True

		elif self.exploration_state == 'SCANNING_QR':
			if self.scan_state == 'SCAN_MID_YAW' or self.scan_state == 'SCAN_LEFT_YAW' or self.scan_state == 'SCAN_RIGHT_YAW':
				shelf = self.shelf_objects_curr

				try:
					name, count = shelf.object_name[0], shelf.object_count[0]
				except:
					self.logger.info("[SCAN RESULT] No QR detected in the current scan.")
					return
				if(name[:2]=="QR"):
					self.qr_code_str = name[3:]
					self.scan_complete = True
					# self.publish_shelf_data()
					return
		
		elif self.find_next_shelf and self.decision_state == 'SCAN':
			if self.scan_state == 'SCAN_MID_YAW' or self.scan_state == 'SCAN_LEFT_YAW' or self.scan_state == 'SCAN_RIGHT_YAW':
				shelf = self.shelf_objects_curr

				try:
					name, count = shelf.object_name[0], shelf.object_count[0]
				except:
					self.logger.info("[SCAN RESULT] No QR detected in the current scan.")
					return
				if(name[:2]=="QR"):
					self.qr_code_str = name[3:]
					self.logger.info(f"Found QR {self.qr_code_str}")
					self.scan_complete = True
					return

	def shelf_data_callback(self, message):
		"""Callback function to handle combined shelf data (objects + QR codes).

		Args:
			message: ROS2 message containing combined shelf data.

		Returns:
			None
		"""
		self.latest_shelf_data = message
		
		return

	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)


	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	def goal_result_callback(self, future):
		"""
		Callback function executed when the navigation goal reaches a final result.

		Args:
			future (rclpy.Future): The future that is result of the navigation action.
		"""
		status = future.result().status
		# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")
			self.goal_failed = False
		else:
			self.logger.warn(f"Goal failed with status: {status}")
			self.goal_failed = False

		self.goal_completed = True  # Mark goal as completed.
		self.goal_handle_curr = None  # Clear goal handle.

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")
		
		self.goal_progress = {
			"distance_remaining": distance_remaining,
			"navigation_time": navigation_time,
			"estimated_time_remaining": estimated_time_remaining,
			"number_of_recoveries": number_of_recoveries
		}

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True

	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q
	
	def quaternion_to_yaw(self, q: Quaternion) -> float:
		"""
		Convert a quaternion (assumed to represent only yaw) to a yaw angle in radians.
		
		Parameters:
			q (Quaternion): Quaternion with x = y = 0

		Returns:
			float: Yaw angle in radians
		"""
		return 2 * math.atan2(q.z, q.w)

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
					   yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()