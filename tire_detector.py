import cv2
import numpy as np
import argparse
import imutils

import time
import os

#python tire_detector.py -f car3.jpg -i image
LABEL = 'tire'
COLOR = (255, 0, 255) # purple
MODEL_PATH = 'model' # folder name

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="path to input image or video")
ap.add_argument("-i", "--input", required=True, help="type of input: 'image' or 'video'")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# LOAD YOLO MODEL
weightsPath = os.path.sep.join([MODEL_PATH, "custom-yolov4-tiny-detector_4000.weights"])
configPath = os.path.sep.join([MODEL_PATH, "custom-yolov4-tiny-detector.cfg"])
print("[INFO] loading YOLO from disk...")
model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# determine only the *output* layer names that we need from YOLO
layer_names = model.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

def draw_predictions(image):
    """
    Draws bounding boxes and confidences on an image or a frame
    :param image:
    :return: image
    """
    (H, W) = image.shape[:2] # input dimensions
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    outputs = model.forward(layer_names)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO inference took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []

    # loop over outputs
    for output in outputs:
    	# loop over each of the detections
    	for detection in output:
    		confidence = detection[5]
    		# filter out predictions below the threshold
    		if confidence > args["confidence"]:
    			# scale the bounding box coordinates back
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
	    		# use the center (x, y)-coordinates to derive the top and
	    		# and left corner of the bounding box
	    		x = int(centerX - (width / 2))
	    		y = int(centerY - (height / 2))
	    		# update our list of bounding box coordinates, confidences,
	    		# and class IDs
	    		boxes.append([x, y, int(width), int(height)])
	    		confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # MAKE THE OUTPUT IMAGE
    if len(idxs) > 0:       # ensure at least one detection exists
    	# loop over the indexes we are keeping
	    for i in idxs.flatten():
	    	# extract the bounding box coordinates
	    	(x, y) = (boxes[i][0], boxes[i][1])
	    	(w, h) = (boxes[i][2], boxes[i][3])
	    	# draw a bounding box rectangle and label on the image
	    	cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)

	    	# text with background
	    	text = "{:.1f} %".format(confidences[i] * 100)
	    	# text = "{}: {:.2f}".format(LABEL, confidences[i])
	    	(text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	    	cv2.rectangle(image, (x - 1, y + 1),
	    				  (x + max(w, text_width) + 1, y - text_height - baseline - 1),
	    				  COLOR, -1)
	    	cv2.putText(image, text,
	    				(x, y - int(0.2 * text_height)),
	    				cv2.FONT_HERSHEY_SIMPLEX,
	    				0.5, (255, 255, 255), 1)
    return image

# LOAD INPUT
if args["input"] == 'image':
    image = cv2.imread(os.path.sep.join(["Input", args["file"]]))
    image = draw_predictions(image)
    cv2.imwrite('Output/{}'.format(args["file"]), image)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

#if args["input"] == 'video':
