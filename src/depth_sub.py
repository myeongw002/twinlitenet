#!/usr/bin/env python3  

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
from twinlitenet.msg import PixelCoordinates
import sensor_msgs.point_cloud2 as pcl2
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
        self.depth_sub = rospy.Subscriber( depth_topic, Image, self.depth_callback, queue_size = 1)     
        self.ll_point_pub = rospy.Publisher("/ll_pointcloud", PointCloud2, queue_size=1)
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
        
        try:        
            ll_xyz_coords, _ = self.cal_XYZ(self.ll_coords, self.depth_image)
            ll_pointcloud_msg = self.xyz_to_pointcloud2(ll_xyz_coords)
            self.ll_point_pub.publish(ll_pointcloud_msg)
        except:
            pass
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
            depth_value = depth_image[y, x]
            
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
        
