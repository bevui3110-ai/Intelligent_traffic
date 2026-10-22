# Import necessary packages
import time
import cv2
import csv
import numpy as np
from tracker import *
from datetime import datetime, timedelta
import CVCascadeFilter
from threading import Timer


# Initialize Tracker
tracker = EuclideanDistTracker()
greenack = 0
redack = 1
redack1 = 1
greenack1 = 0
dt2 = 0
dt1 = 0
# Initialize the videocapture object
cap = cv2.VideoCapture('ind.mp4')
input_size = 320

# Detection confidence threshold
confThreshold = 0.45
nmsThreshold = 0.4

font_color = (52,235,198)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 300
up_line_position = middle_line_position - 10
down_line_position = middle_line_position + 10

# Store Coco Names in a list
classesFile = "YOLO Algorithm files/coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

# Model Files
modelConfiguration = 'YOLO Algorithm files/yolov3-320.cfg'
modelWeigheights = 'YOLO Algorithm files/yolov3.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backenda

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# THÊM MỚI: Cấu hình tối ưu tốc độ (Frame skipping cho VẼ hiển thị)
FRAME_SKIP = 2
frame_count = 0
last_drawn_frame = None

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


def redLight(img):
    global redack
    global redack1
    global greenack
    global greenack1
    global dt2
    global dt1

    if redack == 1:
        if redack1 == 1:
            dt2 = int(("%s" % datetime.now().second))
            redack1 = 0
        dt1 = int(("%s" % datetime.now().second))
        #print("dt1",dt1,dt2)
        # print("dt2",dt2)
        cc = 0
        cc2 = 0
        if dt1 > dt2:
            cc = dt1
            cc2 = dt2
        else:
            cc = dt2
            cc2 = dt1
        if cc < (cc2 + 20):
            cv2.circle(img, (530, 30), 14, (0, 0, 255), -1)
            #cv2.circle(img, (430, 30), 14, (0, 0, 255), -1)
            cv2.circle(img, (430, 65), 14, (0, 255, 0), -1)
            str1 = str(dt1 - dt2)
            cv2.putText(img, str1, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            cv2.putText(img, str1, (450, 69), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            greenack = 0
        else:
            greenack = 1
            greenack1 = 1
            redack = 0


def greenLight(img, density):
    global redack
    global redack1
    global greenack
    global greenack1
    global dt2
    global dt1
    yu = 0
    if density < 10:
        yu = 5
    else:
        yu = 10
    if greenack == 1:
        if greenack1 == 1:
            dt2 = int(("%s" % datetime.now().second))
            greenack1 = 0
        dt1 = int(("%s" % datetime.now().second))
        #print("dt1",dt1, dt2)
        # print("dt2",dt2)
        cc = 0
        cc2 = 0
        if dt1 > dt2:
            cc = dt1
            cc2 = dt2
        else:
            cc = dt2
            cc2 = dt1
        if cc < cc2 + yu:
            cv2.circle(img, (530, 65), 14, (0, 255, 0), -1)
            #cv2.circle(img, (430, 65), 14, (0, 255, 0), -1)
            cv2.circle(img, (430, 30), 14, (0, 0, 255), -1)
            str1 = str(dt1 - dt2)
            cv2.putText(img, str1, (550, 69), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            cv2.putText(img, str1, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            redack = 0
        else:
            redack = 1
            redack1 = 1
            greenack = 0


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]
startTracker = {}  # STORE STARTING TIME OF CARS
endTracker = {}  # STORE ENDING TIME OF CARS
markGap = 20
fpsFactor = 3


# Function for count vehicle
def count_vehicle(density, box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list:
            startTracker[id] = datetime.now()
            temp_up_list.append(id)

    elif (iy < down_line_position) and (iy > middle_line_position):
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            endTracker[id] = datetime.now()
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)


# Function for calculating the speed of vehicle
def estimateSpeed(position):
    timeDiff = int(CVCascadeFilter.CVCascadeFilter.timings(bin(position).replace("0b", "")), 2)
    # print("timeDiff", timeDiff)
    speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
    return speed


# Function for finding the detected objects from the network output
def postProcess(outputs, img, should_draw=True):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        speed = estimateSpeed(i)

        # THÊM MỚI: Chỉ thực hiện vẽ Bounding box khi should_draw == True
        if should_draw:
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}% {speed} Kmph',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object (LUÔN ĐẾM TRÊN MỌI FRAME)
    boxes_ids = tracker.update(detection)
    density = len(boxes_ids)
    for box_id in boxes_ids:
        count_vehicle(density, box_id, img if should_draw else None)

    return density


def realTime():
    global frame_count, last_drawn_frame
    while True:
        success, img = cap.read()
        if not success or img is None:
            break

        # THÊM MỚI: Đếm khung hình để thực hiện Frame Skipping CHỈ CHO VẼ HIỂN THỊ (FRAME_SKIP = 2)
        frame_count += 1
        should_draw = (frame_count % FRAME_SKIP == 0) or (last_drawn_frame is None)

        img = cv2.resize(img, (1280, 720))
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        # LUÔN CHẠY ĐẾM XE VÀ TRACKER TRÊN MỌI KHUNG HÌNH (Full Precision)
        den = postProcess(outputs, img, should_draw=should_draw)

        # THÊM MỚI: Chỉ vẽ các lớp overlay khi should_draw == True
        if should_draw:
            cv2.line(img, (500, 10), (560, 10), (0, 255, 0), 2)
            cv2.line(img, (500, 10), (500, 120), (0, 255, 0), 2)
            cv2.line(img, (560, 10), (560, 120), (0, 255, 0), 2)
            cv2.line(img, (500, 120), (560, 120), (0, 255, 0), 2)
            cv2.circle(img, (530, 30), 15, (0, 0, 0), 2)
            cv2.circle(img, (530, 65), 15, (0, 0, 0), 2)
            cv2.circle(img, (530, 100), 15, (0, 0, 0), 2)
            cv2.line(img, (400, 10), (460, 10), (0, 255, 0), 2)
            cv2.line(img, (400, 10), (400, 120), (0, 255, 0), 2)
            cv2.line(img, (460, 10), (460, 120), (0, 255, 0), 2)
            cv2.line(img, (400, 120), (460, 120), (0, 255, 0), 2)
            cv2.circle(img, (430, 30), 15, (0, 0, 0), 2)
            cv2.circle(img, (430, 65), 15, (0, 0, 0), 2)
            cv2.circle(img, (430, 100), 15, (0, 0, 0), 2)

            redLight(img)
            density = 5 if 0 <= den <= 20 else (10 if 20 <= den <= 40 else 15)
            greenLight(img, density)

            cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
            cv2.putText(img, "Up        ", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.line(img, (0, up_line_position), (iw, up_line_position), (28, 44, 253), 2)
            cv2.putText(img, "Down        ", (40, 340), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.line(img, (0, down_line_position), (iw, down_line_position), (23, 2, 7), 2)

            cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

            last_drawn_frame = img.copy()

        # Hiển thị frame gần nhất đã vẽ
        if last_drawn_frame is not None:
            cv2.imshow('Output', last_drawn_frame)

        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    realTime()
    postProcess()
    count_vehicle()
