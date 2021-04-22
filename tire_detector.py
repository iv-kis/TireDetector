import cv2
import numpy as np
import argparse
import imutils

import time
import os
import platform

MODEL_PATH = 'model'  # folder name
LABEL = 'tire'

COLOR = (255, 0, 255)  # purple
CAP_COLOR = (0, 0, 0)
ALPHA = 0.5
TEXT_SIZE = 0.7
TEXT_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="path to input image or video")
ap.add_argument("-i", "--input", required=True, help="type of input: image / video")
ap.add_argument("-m", "--model", required=True, help="yolo4 / yolo4tiny")
ap.add_argument("-c", "--confidence", type=float, default=0.25, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# LOAD YOLO MODEL
if args["model"]=='yolo4tiny':
    weightsPath = os.path.sep.join([MODEL_PATH, "custom-yolov4-tiny-detector_4000.weights"])
    configPath = os.path.sep.join([MODEL_PATH, "custom-yolov4-tiny-detector.cfg"])
elif args["model"]=='yolo4':
    weightsPath = os.path.sep.join([MODEL_PATH, "custom-yolov4-detector_3000.weights"])
    configPath = os.path.sep.join([MODEL_PATH, "custom-yolov4-detector.cfg"])
else:
    raise Exception("Wrong model name. Must be yolo4 or yolo4tiny")
print("[INFO] loading YOLO from disk...")
model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# determine only the output layer names that we need from YOLO
layer_names = model.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

def draw_predictions(image):
    """
    Draws bounding boxes and confidences on an image or a frame
    :param image:
    :return: image
    """
    (H, W) = image.shape[:2]  # input dimensions
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    outputs = model.forward(layer_names)
    end = time.time()
    # show timing information
    infer_time = end - start

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
    caption_text = ' Model: {}. No objects detected'.format(args['model'])
    if len(idxs) > 0:  # ensure at least one detection exists
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # text with background
            text = "{:.1f}%".format(confidences[i] * 100)
            imboxes = image.copy()  # semi-opaque layer
            cv2.rectangle(imboxes, (x, y), (x + w, y + h), COLOR, 4)

            (text_width, text_height), text_baseline = cv2.getTextSize(text, TEXT_FONT, TEXT_SIZE, TEXT_THICKNESS)
            cv2.rectangle(imboxes, (x - 2, y + h - 2),
                          (x + max(w, text_width) + 2, y + h + text_height + text_baseline),
                          COLOR, -1)
            cv2.addWeighted(imboxes, ALPHA, image, 1-ALPHA, 0, image)
            cv2.putText(image, text,
                        (x, y + h + text_baseline + int(0.8 * text_height)),
                        TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)
        caption_text = " Model: {}. Inference time: {:.1f} ms".format(args["model"], infer_time * 1000)
    # Caption
    imcaption = image.copy()
    (text_width, text_height), text_baseline = cv2.getTextSize(caption_text, TEXT_FONT, TEXT_SIZE, TEXT_THICKNESS)
    cv2.rectangle(imcaption, (0, 0), (W, text_height + 2 * text_baseline), CAP_COLOR, -1)
    cv2.addWeighted(imcaption, ALPHA, image, 1 - ALPHA, 0, image)
    cv2.putText(image, caption_text,
                (0, text_height + text_baseline),
                TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)
        
    return image, infer_time

# Print system info
print(platform.platform())
print(platform.processor())

# LOAD INPUT
if args["input"] == 'image':
    # python tire_detector.py -f car3.jpg -i image -m yolo4
    image = cv2.imread(os.path.sep.join(["Input", args["file"]]))
    image, infer_time = draw_predictions(image)
    print("[INFO] YOLO inference took {:.6f} seconds".format(infer_time))
    cv2.imwrite(os.path.sep.join(['Output','{}_{}'.format(args["model"], args["file"])]), image)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

elif args["input"] == 'video':
    # " Model: {}. Inference time: {:.1f} ms".format(args["model"], infer_time * 1000)
    cap = cv2.VideoCapture(os.path.sep.join(["Input", args["file"]]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    (W, H) = (None, None)
    # Estimation of the processing time
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
    # Processing of the frames
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed: # end of the stream
            break
        # Object detection
        frame, infer_time = draw_predictions(frame)
        # Initialization of writer if None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            writer = cv2.VideoWriter(os.path.sep.join(["Output",
                                                       "{}_{}".format(args["model"], args["file"])
                                                       ]),
                                     fourcc, fps, (frame.shape[1], frame.shape[0]), True
                                     )
            # some information on processing single frame
            if total > 0:
                print("[INFO] single frame took {:.4f} seconds".format(infer_time))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    infer_time * total))
        # write the output frame to disk
        writer.write(frame)
    print("[INFO] cleaning up...")
    writer.release()
    cap.release()
else:
    raise Exception('Wrong input type. Must be image or video')