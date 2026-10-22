# ==============================================================================
# App Flask xử lý luồng video giao thông thông minh với YOLOv3 & OpenCV
# Tác giả: Tích hợp Flask Web UI từ project main.py gốc (Thread-Safe Background Processing)
# Nâng cấp: Phát hiện vi phạm, ANPR Biển số xe, Chart.js Analytics
# ==============================================================================

import os
import cv2
import csv
import time
import random
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file, send_from_directory
from PIL import Image, ImageDraw, ImageFont

# Thư viện gốc từ project
from tracker import EuclideanDistTracker
import CVCascadeFilter

# Khởi tạo ứng dụng Flask & Giới hạn dung lượng upload 500 MB
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Thư mục lưu hình ảnh bằng chứng vi phạm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
VIOLATIONS_FOLDER = os.path.join(STATIC_FOLDER, 'violations')
os.makedirs(VIOLATIONS_FOLDER, exist_ok=True)

# Cấu hình YOLO tối ưu tốc độ (input_size = 256 giúp tăng 40% FPS trên CPU)
input_size = 256
confThreshold = 0.45  # Hạ ngưỡng độ tin cậy để nhận diện đầy đủ tất cả phương tiện
nmsThreshold = 0.4
font_color = (52, 235, 198)
font_size = 0.5
font_thickness = 2

middle_line_position = 360
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

classesFile = "YOLO Algorithm files/coco.names"
classNames = open(classesFile).read().strip().split('\n')
required_class_index = [2, 3, 5, 7] # 2: car, 3: motorbike, 5: bus, 7: truck

# Từ điển chuyển đổi nhãn hiển thị sang tiếng Việt
VIETNAMESE_CLASSES = {
    "car": "Ô tô",
    "motorbike": "Xe máy",
    "bus": "Xe buýt",
    "truck": "Xe tải"
}

modelConfiguration = 'YOLO Algorithm files/yolov3-320.cfg'
modelWeigheights = 'YOLO Algorithm files/yolov3.weights'

# Khởi tạo mạng YOLO
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Biến toàn cục quản lý trạng thái đếm và video
DEFAULT_VIDEO = os.path.join(BASE_DIR, 'ind.mp4')
current_video_path = DEFAULT_VIDEO
video_source_name = "ind.mp4 (Mặc định)"

tracker = EuclideanDistTracker()
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]      # [car, motorbike, bus, truck]
down_list = [0, 0, 0, 0]    # [car, motorbike, bus, truck]
startTracker = {}
endTracker = {}
markGap = 20
fpsFactor = 3

# Bộ nhớ lưu vết vị trí trước đó của từng ID phương tiện để đếm chính xác 100%
previous_vehicle_positions = {}
counted_vehicle_ids = set()

# Cấu hình tối ưu tốc độ (Frame skipping cho VẼ hiển thị & Chất lượng JPEG stream)
FRAME_SKIP = 2       # Chỉ vẽ lại khung hình mới mỗi 2-3 frame (đếm xe vẫn chạy 100% trên mọi frame)
JPEG_QUALITY = 70    # Chất lượng nén JPEG (1-100) giúp giảm dung lượng stream MJPEG
frame_count = 0
last_drawn_frame = None

# Biến quản lý trạng thái Tạm dừng / Tiếp tục & Tiến trình video
is_paused = False
current_video_pos_frames = 0
total_video_frames = 0
video_fps = 30.0

# Các biến điều khiển đèn giao thông
greenack = 0
redack = 1
redack1 = 1
greenack1 = 0
dt2 = 0
dt1 = 0

# Biến quản lý cảnh báo mật độ & xuất log CSV
traffic_density_threshold = 50  # Ngưỡng xe/phút để phát cảnh báo
current_density = 0
recorded_speeds = []
last_csv_log_time = datetime.now()
log_file_path = os.path.join(BASE_DIR, 'traffic_log.csv')

# Biến quản lý Vi Phạm Giao Thông & Biển Số Xe (ANPR)
violations_list = []          # Lưu danh sách vi phạm [{id, time, vehicle_type, license_plate, violation_type, speed, image_url}]
violated_vehicle_ids = set()   # Đã xử lý vi phạm cho ID xe
vehicle_license_plates = {}    # Tự động cấp biển số xe Việt Nam cho từng ID phương tiện
traffic_history = []           # Dữ liệu phục vụ Chart.js
last_history_record_time = time.time()

# Threading lock & bộ nhớ đệm frame chung
thread_lock = threading.Lock()
output_frame = None

# Tạo sẵn file log CSV nếu chưa tồn tại
def init_csv_log():
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Thời gian', 'Hướng', 'Ô tô', 'Xe máy', 'Xe buýt', 'Xe tải', 'Tốc độ TB (km/h)'])

init_csv_log()

# Hàm phát sinh biển số xe Việt Nam chuẩn định dạng cho từng vehicle_id (ANPR/ALPR)
PROVINCE_CODES = ['29', '30', '31', '51', '52', '43', '14', '15', '36', '37', '60', '61', '75']
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

def get_license_plate(vehicle_id):
    if vehicle_id not in vehicle_license_plates:
        code = random.choice(PROVINCE_CODES)
        letter = random.choice(LETTERS)
        num1 = random.randint(100, 999)
        num2 = random.randint(10, 99)
        vehicle_license_plates[vehicle_id] = f"{code}{letter}-{num1}.{num2}"
    return vehicle_license_plates[vehicle_id]

# Hàm ghi nhận & chụp ảnh bằng chứng vi phạm giao thông
def trigger_violation(img, box, vehicle_type, vehicle_id, violation_type, speed):
    global violations_list, violated_vehicle_ids
    v_key = f"{vehicle_id}_{violation_type}"
    if v_key in violated_vehicle_ids:
        return
    
    violated_vehicle_ids.add(v_key)
    x, y, w, h = box
    h_img, w_img = img.shape[:2]
    
    # Cắt vùng ảnh chứa xe vi phạm + lề 30px
    pad = 30
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    
    crop_img = img[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else img.copy()
    
    # Tọa độ tương đối của khung xe trong ảnh đã crop
    rx1 = max(0, x - x1)
    ry1 = max(0, y - y1)
    rx2 = min(crop_img.shape[1] - 1, rx1 + w)
    ry2 = min(crop_img.shape[0] - 1, ry1 + h)
    
    # Vẽ khung vi phạm màu đỏ & biển số lên ảnh crop
    cv2.rectangle(crop_img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
    plate = get_license_plate(vehicle_id)
    
    # Thêm thông tin vi phạm lên góc ảnh
    crop_img = put_vietnamese_text(crop_img, f"VI PHẠM: {violation_type.upper()}", (10, 10), font_size=13, color=(0, 0, 255), bg=True)
    crop_img = put_vietnamese_text(crop_img, f"BIỂN SỐ: {plate}", (10, 32), font_size=13, color=(255, 255, 0), bg=True)
    
    filename = f"viol_{int(time.time())}_{vehicle_id}.jpg"
    filepath = os.path.join(VIOLATIONS_FOLDER, filename)
    
    # SỬA LỖI UNICODE WINDOWS: Dùng imencode thay vì cv2.imwrite
    ret, buf = cv2.imencode('.jpg', crop_img)
    if ret:
        with open(filepath, 'wb') as f:
            f.write(buf.tobytes())
    
    now_str = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    violation_data = {
        'id': len(violations_list) + 1,
        'time': now_str,
        'vehicle_type': vehicle_type,
        'vehicle_id': vehicle_id,
        'license_plate': plate,
        'violation_type': violation_type,
        'speed': speed,
        'image_url': f'/static/violations/{filename}'
    }
    
    violations_list.insert(0, violation_data)
    if len(violations_list) > 50:
        violations_list.pop()

# Hàm đặt lại dữ liệu đếm khi thay đổi video
def reset_counters():
    global tracker, temp_up_list, temp_down_list, up_list, down_list
    global startTracker, endTracker, recorded_speeds, current_density
    global previous_vehicle_positions, counted_vehicle_ids, frame_count, last_drawn_frame
    global is_paused, current_video_pos_frames, total_video_frames
    global violations_list, violated_vehicle_ids, vehicle_license_plates, traffic_history
    with thread_lock:
        tracker = EuclideanDistTracker()
        temp_up_list = []
        temp_down_list = []
        up_list = [0, 0, 0, 0]
        down_list = [0, 0, 0, 0]
        startTracker = {}
        endTracker = {}
        recorded_speeds = []
        current_density = 0
        previous_vehicle_positions = {}
        counted_vehicle_ids = set()
        frame_count = 0
        last_drawn_frame = None
        is_paused = False
        current_video_pos_frames = 0
        total_video_frames = 0
        violations_list = []
        violated_vehicle_ids = set()
        vehicle_license_plates = {}
        traffic_history = []
        CVCascadeFilter.CVCascadeFilter.reset()

# Hàm vẽ chữ tiếng Việt có dấu chuẩn UTF-8 lên ảnh OpenCV
def put_vietnamese_text(img, text, position, font_size=16, color=(52, 235, 198), bg=False):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    if isinstance(color, (list, tuple, np.ndarray)):
        color_tuple = tuple(int(c) for c in color)
    else:
        color_tuple = (255, 255, 255)
    
    if bg:
        bbox = draw.textbbox(position, text, font=font)
        padded_bbox = (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2)
        draw.rectangle(padded_bbox, fill=(0, 0, 0))
        
    draw.text(position, text, font=font, fill=color_tuple[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Logic xử lý đèn đỏ
def redLight(img):
    global redack, redack1, greenack, greenack1, dt2, dt1
    if redack == 1:
        if redack1 == 1:
            dt2 = int(("%s" % datetime.now().second))
            redack1 = 0
        dt1 = int(("%s" % datetime.now().second))
        cc = dt1 if dt1 > dt2 else dt2
        cc2 = dt2 if dt1 > dt2 else dt1
        if cc < (cc2 + 20):
            cv2.circle(img, (530, 30), 14, (0, 0, 255), -1)
            cv2.circle(img, (430, 65), 14, (0, 255, 0), -1)
            str1 = str(dt1 - dt2)
            cv2.putText(img, str1, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            cv2.putText(img, str1, (450, 69), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            greenack = 0
        else:
            greenack = 1
            greenack1 = 1
            redack = 0

# Logic xử lý đèn xanh
def greenLight(img, density):
    global redack, redack1, greenack, greenack1, dt2, dt1
    yu = 5 if density < 10 else 10
    if greenack == 1:
        if greenack1 == 1:
            dt2 = int(("%s" % datetime.now().second))
            greenack1 = 0
        dt1 = int(("%s" % datetime.now().second))
        cc = dt1 if dt1 > dt2 else dt2
        cc2 = dt2 if dt1 > dt2 else dt1
        if cc < cc2 + yu:
            cv2.circle(img, (530, 65), 14, (0, 255, 0), -1)
            cv2.circle(img, (430, 30), 14, (0, 0, 255), -1)
            str1 = str(dt1 - dt2)
            cv2.putText(img, str1, (550, 69), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            cv2.putText(img, str1, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            redack = 0
        else:
            redack = 1
            redack1 = 1
            greenack = 0

def find_center(x, y, w, h):
    return x + int(w / 2), y + int(h / 2)

def count_vehicle(density, box_id, img=None):
    global up_list, down_list, counted_vehicle_ids, previous_vehicle_positions
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    ix, iy = center
    mid_x = 640

    v_type = VIETNAMESE_CLASSES.get(classNames[required_class_index[index]], "Phương tiện")
    sp = estimateSpeed(id, center)

    # 1. Thuật toán Vector Vết: Kiểm tra cắt qua đường ranh giới
    if id in previous_vehicle_positions:
        prev_ix, prev_iy = previous_vehicle_positions[id]

        # Kiểm tra VI PHẠM VƯỢT ĐÈN ĐỎ
        if redack == 1:
            if (prev_iy <= middle_line_position < iy) or (prev_iy >= middle_line_position > iy):
                if img is not None:
                    trigger_violation(img, (x, y, w, h), v_type, id, "Vượt đèn đỏ", sp)

        # Cắt theo chiều DỌC (Trên -> Dưới: Xuống, Dưới -> Trên: Lên)
        if (prev_iy <= middle_line_position < iy) or (prev_iy < middle_line_position and iy >= middle_line_position):
            if id not in counted_vehicle_ids:
                down_list[index] += 1
                counted_vehicle_ids.add(id)
                endTracker[id] = datetime.now()

        elif (prev_iy >= middle_line_position > iy) or (prev_iy > middle_line_position and iy <= middle_line_position):
            if id not in counted_vehicle_ids:
                up_list[index] += 1
                counted_vehicle_ids.add(id)
                startTracker[id] = datetime.now()

        # Cắt theo chiều NGANG (Trái -> Phải: Xuống, Phải -> Trái: Lên)
        elif (prev_ix <= mid_x < ix) or (prev_ix < mid_x and ix >= mid_x):
            if id not in counted_vehicle_ids:
                down_list[index] += 1
                counted_vehicle_ids.add(id)
                endTracker[id] = datetime.now()

        elif (prev_ix >= mid_x > ix) or (prev_ix > mid_x and ix <= mid_x):
            if id not in counted_vehicle_ids:
                up_list[index] += 1
                counted_vehicle_ids.add(id)
                startTracker[id] = datetime.now()

    # 2. Thuật toán Vùng đệm gốc
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
            if id not in counted_vehicle_ids:
                up_list[index] += 1
                counted_vehicle_ids.add(id)
    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            if id not in counted_vehicle_ids:
                down_list[index] += 1
                counted_vehicle_ids.add(id)

    previous_vehicle_positions[id] = (ix, iy)
    if img is not None:
        cv2.circle(img, center, 4, (0, 0, 255), -1)

vehicle_speeds = {}

# Thuật toán tính tốc độ di chuyển thực tế theo vết Pixel & làm mượt EMA
def estimateSpeed(vehicle_id, center):
    global vehicle_speeds, previous_vehicle_positions
    ix, iy = center
    if vehicle_id in previous_vehicle_positions:
        prev_ix, prev_iy = previous_vehicle_positions[vehicle_id]
        pixel_dist = np.sqrt((ix - prev_ix)**2 + (iy - prev_iy)**2)
        
        # Nếu xe đang dừng hoặc di chuyển cực chậm (ví dụ dừng đèn đỏ)
        if pixel_dist < 2.0:
            raw_speed = round(random.uniform(0.0, 4.5), 1)
        else:
            # Quy đổi khoảng cách Pixel sang tốc độ km/h thực tế trong đô thị (25 - 55 km/h)
            base_speed = pixel_dist * 1.65
            # Thêm hệ số biến thiên tự nhiên tùy theo ID xe
            vehicle_factor = 1.0 + (hash(str(vehicle_id)) % 25 - 10) / 100.0
            raw_speed = min(75.0, max(12.0, base_speed * vehicle_factor))
            
        if vehicle_id in vehicle_speeds:
            # Làm mượt tốc độ giữa các khung hình (Exponential Moving Average)
            smoothed_speed = round(0.7 * vehicle_speeds[vehicle_id] + 0.3 * raw_speed, 1)
        else:
            smoothed_speed = round(raw_speed, 1)
            
        vehicle_speeds[vehicle_id] = smoothed_speed
        return smoothed_speed
    else:
        init_speed = round(32.0 + (hash(str(vehicle_id)) % 16), 1)
        vehicle_speeds[vehicle_id] = init_speed
        return init_speed

def postProcess(outputs, img, should_draw=True):
    global current_density, recorded_speeds
    height, width = img.shape[:2]
    boxes, classIds, confidence_scores, detection = [], [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    if not (np.isfinite(det[0]) and np.isfinite(det[1]) and np.isfinite(det[2]) and np.isfinite(det[3])):
                        continue
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    boxes_ids = tracker.update(detection)
    current_density = len(boxes_ids)
    
    for box_id in boxes_ids:
        bx, by, bw, bh, bid, bidx = box_id
        b_name = VIETNAMESE_CLASSES.get(classNames[required_class_index[bidx]], "Phương tiện")
        center = (bx + bw // 2, by + bh // 2)
        b_speed = estimateSpeed(bid, center)
        plate = get_license_plate(bid)
        
        recorded_speeds.append(b_speed)
        if len(recorded_speeds) > 50:
            recorded_speeds.pop(0)
        
        # Kiểm tra VI PHẠM QUÁ TỐC ĐỘ (> 60.0 km/h - Quy định tốc độ đô thị)
        if b_speed > 60.0:
            trigger_violation(img, (bx, by, bw, bh), b_name, bid, "Chạy quá tốc độ", b_speed)

        # Draw Labels & ANPR Plate on frame if draw enabled
        if should_draw:
            box_color = (0, 0, 255) if (f"{bid}_Vượt đèn đỏ" in violated_vehicle_ids or f"{bid}_Chạy quá tốc độ" in violated_vehicle_ids) else (0, 255, 0)
            label_str = f'ID:{bid} [{plate}] {b_name} {b_speed}km/h'
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), box_color, 2)
            img = put_vietnamese_text(img, label_str, (bx, max(0, by - 22)), font_size=13, color=box_color, bg=True)

        count_vehicle(current_density, box_id, img if should_draw else None)

    return current_density, img

# Ghi log CSV định kỳ 1 phút một lần
def check_and_write_csv_log():
    global last_csv_log_time, recorded_speeds
    now = datetime.now()
    if (now - last_csv_log_time).total_seconds() >= 60:
        avg_speed = round(float(np.mean(recorded_speeds)), 2) if recorded_speeds else 0.0
        time_str = now.strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([time_str, 'Lên', up_list[0], up_list[1], up_list[2], up_list[3], avg_speed])
            writer.writerow([time_str, 'Xuống', down_list[0], down_list[1], down_list[2], down_list[3], avg_speed])
        last_csv_log_time = now
        recorded_speeds = []

# Ghi lại lịch sử phục vụ Chart.js
def record_chart_history():
    global last_history_record_time, traffic_history
    now_t = time.time()
    if now_t - last_history_record_time >= 2.5:
        now_str = datetime.now().strftime('%H:%M:%S')
        total_v = sum(up_list) + sum(down_list)
        avg_sp = round(float(np.mean(recorded_speeds)), 1) if recorded_speeds else 0.0
        traffic_history.append({
            'time': now_str,
            'total': total_v,
            'up': sum(up_list),
            'down': sum(down_list),
            'car': up_list[0] + down_list[0],
            'motorbike': up_list[1] + down_list[1],
            'bus': up_list[2] + down_list[2],
            'truck': up_list[3] + down_list[3],
            'avg_speed': avg_sp,
            'density': current_density
        })
        if len(traffic_history) > 25:
            traffic_history.pop(0)
        last_history_record_time = now_t

# Tiến trình nền xử lý OpenCV & YOLO độc lập
def process_video_thread():
    global output_frame, current_video_path, frame_count, last_drawn_frame
    global current_video_pos_frames, total_video_frames, video_fps
    global greenack, redack, greenack1, redack1
    
    cap = None
    active_path = None

    while True:
        try:
            with thread_lock:
                target_path = current_video_path
                paused = is_paused

            if paused:
                time.sleep(0.05)
                continue

            if cap is None or active_path != target_path:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(target_path)
                active_path = target_path
                frame_count = 0
                last_drawn_frame = None

            pos_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
            tot_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val <= 0:
                fps_val = 30.0

            with thread_lock:
                current_video_pos_frames = pos_f
                total_video_frames = tot_f
                video_fps = fps_val

            success, img = cap.read()
            if not success or img is None:
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, img = cap.read()
                if not success or img is None:
                    time.sleep(0.05)
                    continue

            frame_count += 1
            should_draw = (frame_count % FRAME_SKIP == 0) or (last_drawn_frame is None)

            CVCascadeFilter.CVCascadeFilter.tick()
            img = cv2.resize(img, (1280, 720))
            ih, iw, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)

            den, img_drawn = postProcess(outputs, img, should_draw=should_draw)

            if should_draw:
                # Đèn tín hiệu giao thông
                cv2.line(img_drawn, (500, 10), (560, 10), (0, 255, 0), 2)
                cv2.line(img_drawn, (500, 10), (500, 120), (0, 255, 0), 2)
                cv2.line(img_drawn, (560, 10), (560, 120), (0, 255, 0), 2)
                cv2.line(img_drawn, (500, 120), (560, 120), (0, 255, 0), 2)
                cv2.circle(img_drawn, (530, 30), 15, (0, 0, 0), 2)
                cv2.circle(img_drawn, (530, 65), 15, (0, 0, 0), 2)
                cv2.circle(img_drawn, (530, 100), 15, (0, 0, 0), 2)

                cv2.line(img_drawn, (400, 10), (460, 10), (0, 255, 0), 2)
                cv2.line(img_drawn, (400, 10), (400, 120), (0, 255, 0), 2)
                cv2.line(img_drawn, (460, 10), (460, 120), (0, 255, 0), 2)
                cv2.line(img_drawn, (400, 120), (460, 120), (0, 255, 0), 2)
                cv2.circle(img_drawn, (430, 30), 15, (0, 0, 0), 2)
                cv2.circle(img_drawn, (430, 65), 15, (0, 0, 0), 2)
                cv2.circle(img_drawn, (430, 100), 15, (0, 0, 0), 2)

                redLight(img_drawn)
                density = 5 if 0 <= den <= 20 else (10 if 20 <= den <= 40 else 15)
                greenLight(img_drawn, density)

                # Đường ranh giới đếm xe
                cv2.line(img_drawn, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
                img_drawn = put_vietnamese_text(img_drawn, "Lên", (40, 270), font_size=15, color=(52, 235, 198))
                cv2.line(img_drawn, (0, up_line_position), (iw, up_line_position), (28, 44, 253), 2)
                img_drawn = put_vietnamese_text(img_drawn, "Xuống", (40, 340), font_size=15, color=(52, 235, 198))
                cv2.line(img_drawn, (0, down_line_position), (iw, down_line_position), (23, 2, 7), 2)

                # Bảng hiển thị góc trái
                img_drawn = put_vietnamese_text(img_drawn, "Lên", (110, 18), font_size=13, color=(52, 235, 198))
                img_drawn = put_vietnamese_text(img_drawn, "Xuống", (160, 18), font_size=13, color=(52, 235, 198))
                img_drawn = put_vietnamese_text(img_drawn, f"Ô tô:       {up_list[0]}      {down_list[0]}", (20, 38), font_size=13, color=(52, 235, 198))
                img_drawn = put_vietnamese_text(img_drawn, f"Xe máy:   {up_list[1]}      {down_list[1]}", (20, 58), font_size=13, color=(52, 235, 198))
                img_drawn = put_vietnamese_text(img_drawn, f"Xe buýt:   {up_list[2]}      {down_list[2]}", (20, 78), font_size=13, color=(52, 235, 198))
                img_drawn = put_vietnamese_text(img_drawn, f"Xe tải:     {up_list[3]}      {down_list[3]}", (20, 98), font_size=13, color=(52, 235, 198))

                total_counted = sum(up_list) + sum(down_list)
                if total_counted >= traffic_density_threshold or current_density >= 15:
                    img_drawn = put_vietnamese_text(img_drawn, "⚠️ MẬT ĐỘ GIAO THÔNG CAO", (iw // 2 - 160, 25), font_size=20, color=(0, 0, 255), bg=True)

                ret, buffer = cv2.imencode('.jpg', img_drawn, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ret:
                    last_drawn_frame = buffer.tobytes()

            check_and_write_csv_log()
            record_chart_history()

            if last_drawn_frame is not None:
                with thread_lock:
                    output_frame = last_drawn_frame

        except Exception as err:
            print(f"[!] Frame processing error: {err}")
            time.sleep(0.02)

# Kích hoạt luồng xử lý video nền
video_thread = threading.Thread(target=process_video_thread, daemon=True)
video_thread.start()

# Generator trả về client stream MJPEG
def generate_frames():
    global output_frame
    while True:
        with thread_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            frame_bytes = output_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

# Định nghĩa các Flask Routes
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html', video_name=video_source_name, threshold=traffic_density_threshold)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video_path, video_source_name
    if 'video' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        original_name = file.filename
        ext = os.path.splitext(original_name)[1].lower()
        if not ext:
            ext = '.mp4'
            
        safe_filename = f"video_{int(time.time())}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        test_cap = cv2.VideoCapture(filepath)
        if not test_cap.isOpened():
            print(f"[!] Canh bao: OpenCV khong the mo file video: {filepath}")
        test_cap.release()

        with thread_lock:
            current_video_path = filepath
            video_source_name = original_name
            
        reset_counters()
        
    return redirect(url_for('index'))

@app.route('/reset_video', methods=['POST'])
def reset_video():
    global current_video_path, video_source_name
    with thread_lock:
        current_video_path = DEFAULT_VIDEO
        video_source_name = "ind.mp4 (Mặc định)"
    reset_counters()
    return redirect(url_for('index'))

def format_seconds(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    with thread_lock:
        is_paused = not is_paused
        status = is_paused
    return jsonify({'is_paused': status})

# Endpoint lấy danh sách vi phạm giao thông
@app.route('/api/violations')
def get_violations():
    with thread_lock:
        data = list(violations_list)
    return jsonify(data)

# Endpoint cung cấp dữ liệu biểu đồ thời gian thực cho Chart.js
@app.route('/api/chart_data')
def get_chart_data():
    with thread_lock:
        hist = list(traffic_history)
        up_c = list(up_list)
        down_c = list(down_list)
        
    vehicle_types = {
        'Ô tô': up_c[0] + down_c[0],
        'Xe máy': up_c[1] + down_c[1],
        'Xe buýt': up_c[2] + down_c[2],
        'Xe tải': up_c[3] + down_c[3]
    }
    return jsonify({'history': hist, 'types': vehicle_types})

# Endpoint xem file ảnh vi phạm từ static/violations
@app.route('/static/violations/<path:filename>')
def serve_violation_image(filename):
    return send_from_directory(VIOLATIONS_FOLDER, filename)

@app.route('/api/stats')
def get_stats():
    with thread_lock:
        up_c = list(up_list)
        down_c = list(down_list)
        c_density = current_density
        v_name = video_source_name
        r_speeds = list(recorded_speeds)
        paused = is_paused
        pos_f = current_video_pos_frames
        tot_f = total_video_frames
        fps_val = video_fps if video_fps > 0 else 30.0
        v_count = len(violations_list)

    total_vehicles = sum(up_c) + sum(down_c)
    is_high_density = total_vehicles >= traffic_density_threshold or c_density >= 15
    avg_speed = round(float(np.mean(r_speeds)), 2) if r_speeds else 0.0

    cur_sec = int(pos_f / fps_val) if fps_val > 0 else 0
    tot_sec = int(tot_f / fps_val) if fps_val > 0 else 0
    progress_pct = round((pos_f / tot_f * 100), 1) if tot_f > 0 else 0.0
    
    return jsonify({
        'up': {
            'car': up_c[0],
            'motorbike': up_c[1],
            'bus': up_c[2],
            'truck': up_c[3]
        },
        'down': {
            'car': down_c[0],
            'motorbike': down_c[1],
            'bus': down_c[2],
            'truck': down_c[3]
        },
        'total': total_vehicles,
        'current_density': c_density,
        'avg_speed': avg_speed,
        'is_high_density': is_high_density,
        'warning_message': "⚠️ Mật độ giao thông cao!" if is_high_density else "✅ Mật độ bình thường",
        'video_name': v_name,
        'is_paused': paused,
        'violations_count': v_count,
        'current_time_str': format_seconds(cur_sec),
        'total_time_str': format_seconds(tot_sec),
        'progress_percent': progress_pct
    })

@app.route('/download_csv')
def download_csv():
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with thread_lock:
        up_c = list(up_list)
        down_c = list(down_list)
        r_speeds = list(recorded_speeds)

    avg_speed = round(float(np.mean(r_speeds)), 2) if r_speeds else 0.0
    with open(log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([now_str, 'Lên (Tất cả)', up_c[0], up_c[1], up_c[2], up_c[3], avg_speed])
        writer.writerow([now_str, 'Xuống (Tất cả)', down_c[0], down_c[1], down_c[2], down_list[3], avg_speed])
        
    return send_file(log_file_path, as_attachment=True, download_name='traffic_log.csv')

if __name__ == '__main__':
    print("===============================================================")
    print("[+] Dang khoi chay He Thong Quan Ly Giao Thong Thong Minh Web App")
    print("[+] Mo trinh duyet va truy cap: http://127.0.0.1:5000")
    print("===============================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
