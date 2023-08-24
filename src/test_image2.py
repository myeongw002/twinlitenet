#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from twinlitenet.msg import PixelCoordinates
import cv2
import numpy as np
import torch
from model import TwinLite as net
import time

class TwinLiteNet:
    
    def __init__(self, img_size=640, half=True):
        
        self.model_path = rospy.get_param("~weights")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = type('', (), {})()  # Simple empty class for storing options
        self.opt.img_size = img_size
        self.half = half
        self.model = self._initialize_model()
        self.bridge = CvBridge()

        # ROS Initialization
        image_topic = rospy.get_param('~image_tpoic')
        self.image_sub = rospy.Subscriber(image_topic, CompressedImage, self.callback)
        self.da_pub = rospy.Publisher('/drivable_area', PixelCoordinates, queue_size = 1)
        self.ll_pub = rospy.Publisher('/line_lane', PixelCoordinates, queue_size = 1)
        
        
        
    def _initialize_model(self):
        model = net.TwinLiteNet()
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.load_state_dict(torch.load(self.model_path))
        
        if self.half:
            model = model.half()  # Convert the model to half precision
        model.eval()

        # Warm-up
        img = torch.zeros((1, 3, self.opt.img_size, self.opt.img_size), device=self.device)  # init img
        _ = model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            
        return model

    def callback(self, data):
        try:
            # Convert CompressedImage message to OpenCV image using cv_bridge
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return

        processed_img = self.process(cv_image)
        cv2.imshow('result', processed_img)
        cv2.waitKey(1)

    def process(self, img):
        original_height, original_width = img.shape[:2]  # Capture original dimensions

        img_resized = cv2.resize(img, (640, 360))
        img_copy = img_resized.copy()

        img_proc = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_proc = np.ascontiguousarray(img_proc)
        img_proc = torch.from_numpy(img_proc)
        img_proc = torch.unsqueeze(img_proc, 0)  # add a batch dimension
        img_proc = img_proc.cuda().float() / 255.0
        if self.half:
            img_proc = img_proc.half()

        with torch.no_grad():
            img_out = self.model(img_proc)

        x0 = img_out[0]
        x1 = img_out[1]

        _, da_predict = torch.max(x0, 1)
        _, ll_predict = torch.max(x1, 1)

        DA = da_predict.byte().cpu().data.numpy()[0] * 255
        LL = ll_predict.byte().cpu().data.numpy()[0] * 255

        # Resize the DA and LL masks back to original dimensions
        DA_original = cv2.resize(DA, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        LL_original = cv2.resize(LL, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        da_coords = np.array(np.where(DA_original > 100))
        ll_coords = np.array(np.where(LL_original > 100))

        self.coord_publisher(self.da_pub, da_coords)
        self.coord_publisher(self.ll_pub, ll_coords)

        alpha = 0.5  # transparency level

        overlay = img.copy()
        overlay[DA_original > 100] = [0, 255, 0]
        overlay[LL_original > 100] = [0, 0, 255]

        # Use cv2.addWeighted() to overlay the transparent areas onto the original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return img


    def coord_publisher(self,publisher,coords):
        msg = PixelCoordinates()
        msg.x_coords = coords[1]
        msg.y_coords = coords[0]
        publisher.publish(msg)

if __name__ == '__main__':
    rospy.init_node('image_processor')
    twinlitenet = TwinLiteNet()
    rospy.spin()  # Keeps the node running

