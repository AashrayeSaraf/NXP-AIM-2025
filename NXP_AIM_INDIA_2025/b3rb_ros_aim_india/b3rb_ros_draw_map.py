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
from rclpy.parameter import Parameter

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import numpy as np
import math

from nav_msgs.msg import OccupancyGrid

import time

QOS_PROFILE_DEFAULT = 10

plt.ion()
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


class MapVisualizer(Node):
	def __init__(self):
		super().__init__('map_visualizer')
		
		print("MapVisualizer node initialized")
		print("Subscribed to: /map")
		print("Displaying occupancy grid with standard ROS colors:")
		print("  - White: Free space (0)")
		print("  - Black: Occupied space (100)")  
		print("  - Gray: Unknown space (-1)")

		self.subscription_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.map_callback,
			QOS_PROFILE_DEFAULT)

	def map_callback(self, message):
		try:
			plt.clf()
			width = message.info.width
			height = message.info.height

			# Validate map dimensions
			if width <= 0 or height <= 0:
				print(f"Invalid map dimensions: {width}x{height}")
				return

			# Convert the occupancy data to a NumPy array.
			data = np.array(message.data).reshape((height, width))

			# Create an RGB image array using vectorized operations.
			image = np.zeros((height, width, 3), dtype=np.uint8)

			# Standard ROS occupancy grid color convention:
			# 0 = free space (white), 100 = occupied (black), -1 = unknown (gray)
			
			# Free space (0) -> White
			free_mask = (data == 0)
			image[free_mask] = [255, 255, 255]  # White (free)
			
			# Occupied space (100) -> Black  
			occupied_mask = (data == 100)
			image[occupied_mask] = [0, 0, 0]  # Black (occupied)
			
			# Unknown space (-1 or other values) -> Gray
			unknown_mask = (data != 0) & (data != 100)
			image[unknown_mask] = [127, 127, 127]  # Gray (unknown)

			plt.imshow(image)
			plt.title("Occupancy Grid Map")
			plt.gca().invert_yaxis()  # Invert the y axis to match ROS convention
			plt.pause(0.01)
			
		except Exception as e:
			print(f"Error in map_callback: {e}")
			print(f"Map info - Width: {message.info.width}, Height: {message.info.height}, Data length: {len(message.data)}")


def main(args=None):
	rclpy.init(args=args)

	try:
		map_visualizer = MapVisualizer()
		rclpy.spin(map_visualizer)
	except KeyboardInterrupt:
		print("Map visualizer interrupted by user")
	except Exception as e:
		print(f"Error in map visualizer: {e}")
	finally:
		# Destroy the node explicitly
		# (optional - otherwise it will be done automatically
		# when the garbage collector destroys the node object)
		if 'map_visualizer' in locals():
			map_visualizer.destroy_node()
		rclpy.shutdown()
		plt.close('all')  # Close all matplotlib windows


if __name__ == '__main__':
	main()
