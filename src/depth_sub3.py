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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
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
        self.lanes = None
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
        print('All topics ready')
        #self.da_point_pub = rospy.Publisher("/da_pointcloud", PointCloud2, queue_size=1)
        
        
        
        
        
    def ll_callback(self, msg):
        x_array = np.array(msg.x_coords)
        y_array = np.array(msg.y_coords)
        self.ll_coords = np.vstack((x_array, y_array))
        #self.ll_coords = self.downsample_coords(self.ll_coords, step=2)
            # Apply DBSCAN clustering on lane pixel coordinates
        scaler = StandardScaler().fit(self.ll_coords.T)
        scaled_ll_coords = scaler.transform(self.ll_coords.T)
        
        # Apply DBSCAN clustering on lane pixel coordinates
        clustering = DBSCAN(eps=0.2, min_samples=25).fit(scaled_ll_coords)

        # Separate lanes based on DBSCAN clustering results
        self.lanes = {}
        for i, label in enumerate(clustering.labels_):
            if label not in self.lanes:
                self.lanes[label] = {"x": [], "y": []}
            self.lanes[label]["y"].append(self.ll_coords[1][i])
            self.lanes[label]["x"].append(self.ll_coords[0][i])
        
        #print(type(self.lanes))
        # 'lanes' now contains separated lane coordinates
        # You can process each lane separately as needed
        
        unique_labels = np.unique(clustering.labels_)
        number_of_lanes = len(unique_labels[unique_labels != -1])
        rospy.loginfo(f"Number of detected lanes: {number_of_lanes}")

        # Polynomial fitting
        self.poly_lanes = {}
        for label, lane in self.lanes.items():
            if label != -1:  # Exclude the noise label
                y = np.array(lane["y"])
                x = np.array(lane["x"])
                coefficients = np.polyfit(y, x, 3)
                self.poly_lanes[label] = coefficients


    def img_callback(self, msg):
        # 기존 코드에서 이미지를 가져옵니다.
        self.image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
        closing = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel, iterations=1)
        if len(closing.shape) == 3:
            closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
        # 연결 성분 분석으로 작은 노이즈 제거
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 100:  # min area threshold
                self.image[labels == i] = 0

        # 클러스터링 된 차선을 그립니다.
        clustering_color = [0, 255, 0]  # 초록색
        for label, coords in self.lanes.items():
            for x, y in zip(coords["x"], coords["y"]):
                # x, y 좌표가 이미지 경계 내에 있는지 확인합니다.
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    self.image[y, x] = clustering_color

        # 라인 피팅된 차선을 그립니다.
        for label, coeffs in list(self.poly_lanes.items()):
            for y in range(self.image.shape[0]):
                x = int(coeffs[0] * y**2 + coeffs[1] * y + coeffs[2])
                # x 좌표가 이미지 경계 내에 있는지 확인합니다.
                if 0 <= x < self.image.shape[1]:
                    self.image[y, x] = [0, 0, 255]  # 빨간색으로 차선을 그립니다.

        # 결과 이미지를 출력합니다.
        cv2.imshow('lane', self.image)
        cv2.waitKey(1)

    def downsample_coords(self,coords, step=2):
        """
        Downsample the coordinates by a given step.
        :param coords: Original coordinates.
        :param step: Step for downsampling.
        :return: Downsampled coordinates.
        """
        return coords[:, ::step]

    def draw_lanes_on_image(self,lanes):
        # Create an empty image
        img = np.zeros((540, 960), dtype=np.uint8)

        # Draw all lanes on the image
        for _, coords in lanes.items():
            for x, y in zip(coords["x"], coords["y"]):
                if 0 <= x < 960 and 0 <= y < 540:  # Ensure the coordinates are within the image boundaries
                    img[int(y), int(x)] = 255

        return img



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
        


            
        
        
if __name__ == '__main__':
    rospy.init_node('pixel_sub')
    pixel_sub = Pixel_Sub()
    rospy.spin()        
        
