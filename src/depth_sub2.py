#!/usr/bin/env python3  

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from twinlitenet.msg import PixelCoordinates
import sensor_msgs.point_cloud2 as pcl2
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np




class Pixel_Sub:
    def __init__(self):
    
        self.K = np.array([684.1913452148438, 0.0, 482.5340881347656, 
                           0.0, 684.1913452148438, 255.77565002441406, 
                           0.0, 0.0, 1.0]).reshape((3,3))             
        self.cv_bridge = CvBridge()                   
        self.ll_coords = None
        self.da_coords = None
        self.polynomial = None
        self.end_point = None
        self.ros_topic_func()
        
        
        
    def ros_topic_func(self):            
        self.ll_sub = rospy.Subscriber('/line_lane', PixelCoordinates, self.ll_callback, queue_size = 1)
        self.ds_sub = rospy.Subscriber('/drivable_area', PixelCoordinates, self.da_callback, queue_size = 1)
        
        depth_topic = rospy.get_param('~depth_topic', default = '/zed/zed_node/depth/depth_registered')
        rospy.wait_for_message('/line_lane', PixelCoordinates)
        rospy.wait_for_message(depth_topic, Image)        
        self.depth_sub = rospy.Subscriber( depth_topic, Image, self.depth_callback, queue_size = 1)    
        self.img_sub = rospy.Subscriber('/zed/zed_node/rgb/image_rect_color/compressed', CompressedImage, self.img_callback, queue_size = 1)
        self.lane_marker_pub = rospy.Publisher("/lane_markers", MarkerArray, queue_size=1) 
        self.ll_point_pub = rospy.Publisher("/ll_pointcloud", PointCloud2, queue_size=1)
        self.poly_point_pub = rospy.Publisher("/poly_pointcloud", PointCloud2, queue_size=1)
        print('All topics ready')
        #self.da_point_pub = rospy.Publisher("/da_pointcloud", PointCloud2, queue_size=1)
        
        
        
        
        
    def ll_callback(self,msg):
        x_array = np.array(msg.x_coords)
        y_array = np.array(msg.y_coords)
        self.ll_coords = np.vstack((x_array, y_array))
        #print(self.ll_coords)
        self.polynomial, self.end_point = self.separate_and_filter_lanes(self.ll_coords)
        #print('end', end_point)

        #print(self.polynomial)    

        
    def da_callback(self,msg):
        x_array = np.array(msg.x_coords)
        y_array = np.array(msg.y_coords)
        self.da_coords = np.vstack((x_array, y_array))
        
        
        
    def img_callback(self,msg):
        self.image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        if self.polynomial:
            lane_img = self.image.copy()
            
            for poly, (start, end) in zip(self.polynomial, self.end_point):
                lane_img = self.draw_polynomial_within_range(lane_img, poly, start[0], end[0])
            
            cv2.imshow('Lanes on Image', lane_img)
            cv2.waitKey(1)        
                
        
        
    def depth_callback(self,msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        #try:        
        ll_xyz_coords, valid_coords = self.cal_XYZ(self.ll_coords, self.depth_image)
        ll_pointcloud_msg = self.xyz_to_pointcloud2(ll_xyz_coords)
        self.ll_point_pub.publish(ll_pointcloud_msg)
   
        #except Exception as e:
         #   rospy.logerr(f"Error processing depth callback: {e}")
        '''
        poly_xyz = self.polynomial_to_XYZ(self.polynomial,self.depth_image)
        poly_pointcloud_msg = self.xyz_to_pointcloud2(ll_xyz_coords)
        self.poly_point_pub.publish(poly_pointcloud_msg)            
        '''    
        '''
        try:        
            da_xyz_coords, _ = self.cal_XYZ(self.da_coords, self.depth_image)
            da_pointcloud_msg = self.xyz_to_pointcloud2(da_xyz_coords)
            self.da_point_pub.publish(da_pointcloud_msg)
        except:
            pass        
        '''




    def draw_polynomial_within_range(self,image, polynomial, x_start, x_end, color=(0, 255, 0), thickness=2):
        """
        Draw the polynomial on the given image within the specified x range.

        Parameters:
        - image: The image on which to draw the polynomial.
        - polynomial: np.poly1d object representing the lane.
        - x_start: Starting x-coordinate for the polynomial.
        - x_end: Ending x-coordinate for the polynomial.
        - color: Color of the polynomial line.
        - thickness: Thickness of the polynomial line.

        Returns:
        - Image with the polynomial drawn.
        """      
        # x의 범위를 생성합니다.
        x_vals = np.linspace(x_start, x_end, num=5)
        
        # 각 x에 대한 y값을 계산합니다.
        y_vals = polynomial(x_vals)
        
        # 이미지 내부의 점만 선택합니다.
        
        valid_points_mask = (y_vals >= 0) & (y_vals <= image.shape[0])
        valid_x_vals = x_vals[valid_points_mask]
        valid_y_vals = y_vals[valid_points_mask]
        
        for i in range(1, len(valid_x_vals)):
            pt1 = (int(valid_x_vals[i-1]), int(valid_y_vals[i-1]))
            pt2 = (int(valid_x_vals[i]), int(valid_y_vals[i]))
            cv2.line(image, pt1, pt2, color, thickness)
            
        return image




    def get_points_on_polynomial(self,coords, polynomial, image_height, image_width, threshold=5):
        """
        Get both the bottom-most point and the highest point on the polynomial.

        Parameters:
        - coords: 2D numpy array with x and y coordinates of points.
        - polynomial: np.poly1d object representing the lane.
        - image_height: The height of the image.
        - image_width: The width of the image.
        - threshold: distance threshold to consider a point as lying on the polynomial.

        Returns:
        - bottom_point: (x, y) tuple representing the bottom-most point on the polynomial.
        - highest_point: (x, y) tuple representing the highest point on the polynomial.
        """
        
        # For bottom point
        x_vals = np.linspace(0, image_width-1, num=5)
        y_vals = polynomial(x_vals)
        
        valid_points_mask = (y_vals >= 0) & (y_vals <= image_height)
        valid_x_vals = x_vals[valid_points_mask]
        valid_y_vals = y_vals[valid_points_mask]
        
        if len(valid_y_vals) == 0:
            bottom_point = None
        else:
            index = np.argmax(valid_y_vals)
            bottom_point = (valid_x_vals[index], valid_y_vals[index])

        # For highest point
        y_pred = polynomial(coords[0])
        mask = coords[1] < y_pred - threshold
        points_above_poly = coords[:, mask]
        
        if points_above_poly.shape[1] == 0:
            highest_point = None
        else:
            min_y_index = np.argmin(points_above_poly[1])
            highest_point = (points_above_poly[0][min_y_index], points_above_poly[1][min_y_index])

        return bottom_point, highest_point


                
                              
    def cal_XYZ(self, coords, depth_image):
        depth_pixel_array = []
        valid_coords = []  # To store coordinates that have valid depth values

        for x, y in zip(coords[0], coords[1]):
            depth_value = depth_image[int(y), int(x)]
            
            if not (np.isnan(depth_value) or np.isinf(depth_value)):
                depth_pixel_array.append(depth_value)
                valid_coords.append([x, y])
        
        valid_coords = np.array(valid_coords).T  # Transpose to get x and y rows
        
        if len(depth_pixel_array) == 0:
            return np.array([]), np.array([])  # Return empty arrays if no valid depths are found
        
        # Convert to homogeneous coordinates
        homo = np.linalg.inv(self.K) @ np.vstack((valid_coords, np.ones(valid_coords.shape[1])))
        
        Z = np.array(depth_pixel_array)
        X = homo[0,:] * Z
        Y = homo[1,:] * Z
        
        XYZ = np.vstack((X, Y, Z)).T
        #print(len(XYZ))
        return XYZ, valid_coords
  


    def xyz_to_pointcloud2(self,xyz_array):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Change this to your desired frame
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('z', 4, PointField.FLOAT32, 1),
                  PointField('y', 8, PointField.FLOAT32, 1)]
        pc2_msg = pcl2.create_cloud(header, fields, xyz_array)
        
        return pc2_msg        
        

    def polynomial_to_XYZ(self, polynomials, depth_image):
        all_xyz_coords = []

        # 이미지의 너비에 대한 x 좌표 범위 생성
        x_range = np.arange(0, depth_image.shape[1])

        for poly in polynomials:
            y_coords = poly(x_range)
            
            # y 좌표의 값을 이미지의 높이로 제한
            y_coords = np.clip(y_coords, 0, depth_image.shape[0]-1)
            
            # 현재 다항식에 대한 (x, y) 좌표 쌍 생성
            coords = np.vstack((x_range, y_coords))
            xyz_coords, _ = self.cal_XYZ(coords, depth_image)
            all_xyz_coords.extend(xyz_coords)

        return np.array(all_xyz_coords)
        
        
        
    def separate_and_filter_lanes(self, ll_coords, degree=2, min_points=100, threshold=0.2):
        """
        Separate lanes, fit them using polynomial of given degree, filter out horizontal lanes, and calculate endpoints.
        
        Parameters:
        - ll_coords: pixel coordinates of the lanes.
        - degree: degree of the polynomial for fitting.
        - min_points: minimum number of points needed to apply regression.
        - threshold: Minimum slope below which lanes are considered horizontal.
        
        Returns:
        - Filtered list of polynomials.
        - Endpoints of the filtered polynomials.
        """
        
        # Separate and fit lanes
        X = np.array(ll_coords).T
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=0.2, min_samples=100).fit(X)
        labels = db.labels_

        unique_labels = np.unique(labels)
        fitted_polynomials = []

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            lane_coords = ll_coords[:, labels == label]
            if len(lane_coords[0]) > min_points:
                p = np.polyfit(lane_coords[0], lane_coords[1], degree)
                fitted_polynomials.append(np.poly1d(p))
        
        # Filter horizontal lanes
        left_lanes = []
        right_lanes = []
        
        for poly in fitted_polynomials:
            slope = poly.deriv()(0)
            if abs(slope) > threshold:
                if slope > 0:  # Left lane
                    left_lanes.append(poly)
                else:  # Right lane
                    right_lanes.append(poly)

        image_center_x = self.image.shape[1] / 2
        bottom_y = self.image.shape[0]

        if left_lanes:
            left_lane = min(left_lanes, key=lambda poly: abs(poly(bottom_y) - image_center_x))
        else:
            left_lane = None

        if right_lanes:
            right_lane = min(right_lanes, key=lambda poly: abs(poly(bottom_y) - image_center_x))
        else:
            right_lane = None

        # Filter out the polynomials
        filtered_polynomials = [lane for lane in [left_lane, right_lane] if lane is not None]
        
        # Calculate endpoints for the filtered polynomials
        lane_endpoints = []
        x_min, x_max = np.min(ll_coords[0]), np.max(ll_coords[0])
        
        for poly in filtered_polynomials:
            y_min = poly(x_min)
            y_max = poly(x_max)
            lane_endpoints.append(((x_min, y_min), (x_max, y_max)))

        return filtered_polynomials, lane_endpoints
            
  

        
if __name__ == '__main__':
    rospy.init_node('pixel_sub')
    pixel_sub = Pixel_Sub()
    rospy.spin()        
        
