# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
from mail import sendEmail
import argparse
import sys
from datetime import datetime,timedelta
import threading
import anvil.users
import anvil.email
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server
anvil.server.connect("XXXXXXXXXX")

# Set up camera constants
IM_WIDTH = 640
IM_HEIGHT = 480

#interval, snap_count, frame_counter, mobile, mailid, duration, endTime

@anvil.server.callable
def argument_pass(email,number,dur):
  mailid = email
  mobile = number
  duration = dur
  print(mailid,mobile,duration)
  main(mailid,mobile,duration)

camera_type = 'picamera'

sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

def processFrame(image):
    # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    im_height, im_width,_ = image.shape
    boxes_list = [None for i in range(boxes.shape[1])]
    for i in range(boxes.shape[1]):
        boxes_list[i] = (int(boxes[0,i,0] * im_height),
                    int(boxes[0,i,1]*im_width),
                    int(boxes[0,i,2] * im_height),
                    int(boxes[0,i,3]*im_width))

    return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize camera and perform object detection.

threshold = 0.4

### Picamera ###
def main(mailid,mobile,duration):
    endTime = datetime.now() + timedelta(minutes=duration)
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH, IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
    rawCapture.truncate(0)
    
    interval=60
    snap_count=-(interval)
    frame_counter=0
    
    for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        if datetime.now() >= endTime:
            break
        frame = np.copy(frame1.array)
        frame.setflags(write=1)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = processFrame(frame)

        # Draw the results of the detection (aka 'visulaize the results')
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                if((frame_counter-snap_count)>=interval):
                    currentTime = datetime.now().strftime('%d/%m/%Y %H:%M')
                    snap_count= frame_counter
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    t = threading.Thread(target=sendEmail, args=(jpeg.tobytes(),currentTime,mailid))
                    t.start()
                    import requests
                    url = "https://www.fast2sms.com/dev/bulk"
                    payload = "sender_id=FSTSMS&message=Alert! Intruder has been Detected! at Time: "+currentTime+"&language=english&route=p&numbers=" + str(
                        mobile)
                    headers = {
                        'authorization': "XXXXXXXX",
                        'Content-Type': "application/x-www-form-urlencoded",
                        'Cache-Control': "no-cache",
                    }
                    requests.request("POST", url, data=payload, headers=headers)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        frame_counter+=1
    
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()
    cv2.destroyAllWindows()
anvil.server.wait_forever()
