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
        
        self.ros_topic_func()
        
        
        
    def ros_topic_func(self):            
        self.ll_sub = rospy.Subscriber('/line_lane', PixelCoordinates, self.ll_callback, queue_size = 1)
        self.ds_sub = rospy.Subscriber('/drivable_area', PixelCoordinates, self.da_callback, queue_size = 1)
        
        depth_topic = rospy.get_param('~depth_topic', default = '/zed/zed_node/depth/depth_registered')
        rospy.wait_for_message('/line_lane', PixelCoordinates)
        rospy.wait_for_message(depth_topic, Image)        
        self.depth_sub = rospy.Subscriber( depth_topic, Image, self.depth_callback, queue_size = 1)    
        self.lane_marker_pub = rospy.Publisher("/lane_markers", MarkerArray, queue_size=1) 
        self.ll_point_pub = rospy.Publisher("/ll_pointcloud", PointCloud2, queue_size=1)
        print('All topics ready')
        #self.da_point_pub = rospy.Publisher("/da_pointcloud", PointCloud2, queue_size=1)
        
        
        
        
        
    def ll_callback(self,msg):
        x_array = np.array(msg.x_coords)
        y_array = np.array(msg.y_coords)
        self.ll_coords = np.vstack((x_array, y_array))
        #print(self.ll_coords)
        
    def da_callback(self,msg):
        x_array = np.array(msg.x_coords)
        y_array = np.array(msg.y_coords)
        self.da_coords = np.vstack((x_array, y_array))
        
    
    def depth_callback(self,msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        #try:        
        ll_xyz_coords, valid_coords = self.cal_XYZ(self.ll_coords, self.depth_image)
        ll_pointcloud_msg = self.xyz_to_pointcloud2(ll_xyz_coords)
        self.ll_point_pub.publish(ll_pointcloud_msg)


        left_regressor, right_regressor = self.find_lanes_with_regression(valid_coords, self.depth_image)
        
        if left_regressor is None or right_regressor is None:
        # Handle the case where the regression could not be performed due to lack of data
            return
            # 모델을 사용하여 예측 (예: 차선의 시작과 끝 지점)
        left_start = left_regressor.predict([[0]])
        left_end = left_regressor.predict([[self.depth_image.shape[1]]])

        right_start = right_regressor.predict([[0]])
        right_end = right_regressor.predict([[self.depth_image.shape[1]]])            
        
        print(left_start,left_end)        
        markers = self.create_lane_markers(left_regressor, right_regressor, msg.width)
        self.lane_marker_pub.publish(markers)    
        #except Exception as e:
         #   rospy.logerr(f"Error processing depth callback: {e}")
            
            
            
        '''
        try:        
            da_xyz_coords, _ = self.cal_XYZ(self.da_coords, self.depth_image)
            da_pointcloud_msg = self.xyz_to_pointcloud2(da_xyz_coords)
            self.da_point_pub.publish(da_pointcloud_msg)
        except:
            pass        
        '''
        
                
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
        


    def find_lanes_with_regression(self, coords, depth_image):
        X = coords[0].reshape(-1, 1)  # X-coordinates
        Y = coords[1].reshape(-1, 1)  # Y-coordinates

        # X 좌표의 중간 값을 기준으로 왼쪽과 오른쪽 데이터를 분류
        midpoint = np.median(X)
        left_lane_indices = X < midpoint
        right_lane_indices = X >= midpoint

        left_X = X[left_lane_indices].reshape(-1, 1)
        left_Y = Y[left_lane_indices].reshape(-1, 1)

        right_X = X[right_lane_indices].reshape(-1, 1)
        right_Y = Y[right_lane_indices].reshape(-1, 1)

        # 왼쪽 차선에 대한 선형 회귀
        if len(left_X) > 0 and len(left_Y) > 0:  # Ensure there's data to fit
            left_regressor = LinearRegression().fit(left_X, left_Y)
        else:
            left_regressor = None

        # 오른쪽 차선에 대한 선형 회귀
        if len(right_X) > 0 and len(right_Y) > 0:  # Ensure there's data to fit
            right_regressor = LinearRegression().fit(right_X, right_Y)
        else:
            right_regressor = None

        return left_regressor, right_regressor




    def create_lane_markers(self, left_regressor, right_regressor, depth_image_width):
        markers = MarkerArray()

        # Create marker for the left lane
        left_marker = Marker()
        left_marker.header.frame_id = "base_link"
        left_marker.type = Marker.LINE_STRIP
        left_marker.action = Marker.ADD
        left_marker.id = 0
        left_marker.scale.x = 0.1
        left_marker.color.a = 1.0
        left_marker.color.r = 1.0

        left_start = left_regressor.predict([[0]])[0][0]
        left_end = left_regressor.predict([[depth_image_width]])[0][0]
        left_marker.points.append(Point(0, left_start, 0))
        left_marker.points.append(Point(depth_image_width, left_end, 0))

        markers.markers.append(left_marker)

        # Create marker for the right lane
        right_marker = Marker()
        right_marker.header.frame_id = "base_link"
        right_marker.type = Marker.LINE_STRIP
        right_marker.action = Marker.ADD
        right_marker.id = 1
        right_marker.scale.x = 0.1
        right_marker.color.a = 1.0
        right_marker.color.g = 1.0

        right_start = right_regressor.predict([[0]])[0][0]
        right_end = right_regressor.predict([[depth_image_width]])[0][0]
        right_marker.points.append(Point(0, right_start, 0))
        right_marker.points.append(Point(depth_image_width, right_end, 0))

        markers.markers.append(right_marker)

        return markers

            
        
        
if __name__ == '__main__':
    rospy.init_node('pixel_sub')
    pixel_sub = Pixel_Sub()
    rospy.spin()        
        
