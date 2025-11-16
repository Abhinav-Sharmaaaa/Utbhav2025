import cv2
import numpy as np
import time
import threading
import queue
from flask import Flask, render_template, Response, jsonify, redirect, url_for
import winsound
import os
from ultralytics import YOLO
import base64
from datetime import datetime

# --- 1. Flask & Threading Setup ---
output_frame = None
frame_lock = threading.Lock()
global_logs = []
log_lock = threading.Lock()
global_captures = []
captures_lock = threading.Lock()

scanning_active = False
video_thread = None
stop_event = None
scanning_lock = threading.Lock()

# Configure Flask
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(os.path.dirname(base_dir), 'frontend')
app = Flask(__name__,
            template_folder=frontend_dir,
            static_folder=frontend_dir,
            static_url_path='')

# --- 2. Constants and Configuration ---
RESIZE_WIDTH = 640

# --- YOLOv8 Config ---
MODEL_FILE_V8_POSE = "yolov8n-pose.pt"
MODEL_FILE_V8_OBJ = "yolov8s.pt"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
DANGEROUS_OBJECTS = ["knife", "scissors", "baseball bat", "bottle", "cell phone"]

# --- [NEW] Motion & Tracking Config ---
PERSISTENCE_TIME = 0.5  # Seconds to keep a box on screen after it disappears
MIN_MOTION_AREA = 100   # Minimum size (in pixels) to be considered "motion"
OVERLAP_THRESHOLD = 0.1 # 10% of the object's box must overlap with motion

# Keypoints for fall detection
BODY_PARTS = {
    "L-shoulder": 5, "R-shoulder": 6, "L-hip": 11, "R-hip": 12
}

# --- Colors ---
COLOR_THREAT_ACTIVE = (0, 0, 255)
COLOR_THREAT_NOTICE = (255, 255, 0)
COLOR_PERSON_SAFE = (0, 255, 0)
COLOR_FPS = (255, 255, 255)

# --- Sound ---
WARNING_SOUND_FILE = "warning.wav"
sound_file_exists = os.path.exists(WARNING_SOUND_FILE)
if not sound_file_exists:
    print(f"Warning: Sound file '{WARNING_SOUND_FILE}' not found.")

# --- 3. Load Models ---
print("Loading models...")
try:
    model_yolo_v8_pose = YOLO(MODEL_FILE_V8_POSE)
    print("YOLOv8-Pose loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8-Pose model: {e}")
    exit()
try:
    model_yolo_v8_obj = YOLO(MODEL_FILE_V8_OBJ)
    classes_v8_obj = model_yolo_v8_obj.names
    print("YOLOv8-Object loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8-Object model: {e}")
    print(f"Please make sure you have '{MODEL_FILE_V8_OBJ}' in the same directory.")
    exit()


# --- 4. Helper Function for Fall Detection ---
def check_for_fall(keypoints_xy):
    try:
        l_shoulder_y = keypoints_xy[BODY_PARTS["L-shoulder"]][1]
        r_shoulder_y = keypoints_xy[BODY_PARTS["R-shoulder"]][1]
        l_hip_y = keypoints_xy[BODY_PARTS["L-hip"]][1]
        r_hip_y = keypoints_xy[BODY_PARTS["R-hip"]][1]
        if l_shoulder_y > l_hip_y + 10 or r_shoulder_y > r_hip_y + 10:
            return True
    except Exception:
        return False
    return False


# --- 5. [REMOVED] MotionTracker class is gone ---


# --- 6. Worker Thread Function ---
# --- [MERGED] Persistence + Background Subtraction ---
def model_worker(frame_queue, result_queue, stop_event_arg):
    global classes_v8_obj

    # --- [NEW] Create Background Subtractor ---
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # --- [NEW] Persistence logic variables ---
    last_known_boxes = {} # Stores the last good box for each label
    last_seen_time = {}   # Stores the last time we saw that label

    while not stop_event_arg.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        current_time = time.time()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- [NEW] 1. Apply Background Subtraction ---
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fgMask = backSub.apply(gray_blur)
        fgMask = cv2.erode(fgMask, None, iterations=2)
        fgMask = cv2.dilate(fgMask, None, iterations=2)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        for c in contours:
            if cv2.contourArea(c) > MIN_MOTION_AREA:
                motion_areas.append(c)

        current_state = {"boxes": [], "people": [], "is_threat": False}
        is_threat = False

        # --- [NEW] 2. Apply YOLO Object Detection ---
        results_obj_list = model_yolo_v8_obj(frame, verbose=False, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)
        
        detected_objects_this_frame = {}
        if results_obj_list:
            result_obj = results_obj_list[0]
            if result_obj.boxes is not None:
                for i in range(len(result_obj.boxes)):
                    box_xywh = result_obj.boxes.xywh[i].cpu().numpy()
                    class_id = int(result_obj.boxes.cls[i].cpu().numpy())
                    label = classes_v8_obj[class_id]

                    if label in DANGEROUS_OBJECTS:
                        confidence = float(result_obj.boxes.conf[i].cpu().numpy())
                        x = int(box_xywh[0] - box_xywh[2] / 2)
                        y = int(box_xywh[1] - box_xywh[3] / 2)
                        w_box = int(box_xywh[2])
                        h_box = int(box_xywh[3])
                        
                        if label not in detected_objects_this_frame:
                           detected_objects_this_frame[label] = {'box': (x, y, w_box, h_box), 'confidence': confidence}

        # --- [NEW] 3. Main Tracking & Motion Check Loop ---
        for label in DANGEROUS_OBJECTS:
            box_to_process = None
            confidence = 0.0

            if label in detected_objects_this_frame:
                # --- Case 1: YOLO found it this frame ---
                box_to_process = detected_objects_this_frame[label]['box']
                confidence = detected_objects_this_frame[label]['confidence']
                last_known_boxes[label] = box_to_process # Update last known position
                last_seen_time[label] = current_time

            elif label in last_known_boxes and (current_time - last_seen_time.get(label, 0)) < PERSISTENCE_TIME:
                # --- Case 2: YOLO missed it. Persist! ---
                box_to_process = last_known_boxes[label] # Re-use the old box
                # confidence remains 0.0

            # --- If we have a box (new or persisted), check it for motion ---
            if box_to_process:
                x, y, w_box, h_box = box_to_process
                threat_level = 1  # Default to "Notice"
                is_moving = False
                object_box_area = w_box * h_box

                for c in motion_areas:
                    (mx, my, mw, mh) = cv2.boundingRect(c)
                    intersect_x1 = max(x, mx)
                    intersect_y1 = max(y, my)
                    intersect_x2 = min(x + w_box, mx + mw)
                    intersect_y2 = min(y + h_box, my + mh)

                    intersect_w = max(0, intersect_x2 - intersect_x1)
                    intersect_h = max(0, intersect_y2 - intersect_y1)
                    intersection_area = intersect_w * intersect_h

                    if object_box_area > 0 and (intersection_area / object_box_area) > OVERLAP_THRESHOLD:
                        is_moving = True
                        break

                if is_moving:
                    threat_level = 2
                    is_threat = True
                
                current_state["boxes"].append((box_to_process, label, confidence, threat_level))

        # --- B: YOLOv8 Pose Detection (Unchanged) ---
        results_v8_list = model_yolo_v8_pose(frame, verbose=False, classes=[0])
        if results_v8_list:
            result = results_v8_list[0]
            if result.boxes is not None and result.keypoints is not None:
                num_people = len(result.boxes)
                for i in range(num_people):
                    box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    keypoints_xy = result.keypoints.xy[i].cpu().numpy()
                    is_falling = check_for_fall(keypoints_xy)
                    if is_falling:
                        is_threat = True
                    current_state["people"].append((box, keypoints_xy, is_falling))

        # --- C: Finalize State ---
        current_state["is_threat"] = is_threat
        try:
            result_queue.put(current_state, block=False)
        except queue.Full:
            pass
        

# --- 7. Drawing Function ---
# --- [RESTORED] Includes transparency for "ghost" boxes ---
def draw_overlays(frame, state, scale):
    for (box, label, confidence, threat_level) in state["boxes"]:
        x, y, w, h = [int(v * scale) for v in box]
        
        # --- [NEW] Make persisted boxes slightly transparent ---
        alpha = 1.0 if confidence > 0 else 0.6 # Full opacity for new, 60% for ghost
        
        if threat_level == 2:
            color = COLOR_THREAT_ACTIVE
            text = f"THREAT: {label.upper()} (ACTIVE)"
            box_thickness = 3
        else: # threat_level == 1
            color = COLOR_THREAT_NOTICE
            text = f"NOTICE: {label} (Carrying)"
            box_thickness = 2
        
        # Draw the box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, box_thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the text (always full opacity)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    for (box, keypoints_xy, is_falling) in state["people"]:
        box_s = [int(v * scale) for v in box]
        keypoints_s = [(int(p[0] * scale), int(p[1] * scale)) for p in keypoints_xy]
        color = COLOR_THREAT_ACTIVE if is_falling else COLOR_PERSON_SAFE
        x1, y1, x2, y2 = box_s
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = "THREAT: FALL DETECTED" if is_falling else "Person"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for (x, y) in keypoints_s:
            cv2.circle(frame, (x, y), 3, color, -1)
            
    return frame

# --- Function to save captures ---
def save_capture(frame, logs):
    # (This function is unchanged)
    global global_captures, captures_lock
    description = "Threat Detected"
    for log in logs:
        if log.startswith("THREAT"):
            description = log
            break
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    capture_data = {
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "image": img_base64
    }
    with captures_lock:
        global_captures.insert(0, capture_data)


# --- 8. Main Video Loop (Threaded Function) ---
def start_video_processing(stop_event_arg):
    # (This function is unchanged)
    global output_frame, frame_lock, global_logs, log_lock, sound_file_exists
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    worker = threading.Thread(target=model_worker, args=(frame_queue, result_queue, stop_event_arg))
    worker.daemon = True
    worker.start()
    last_detection_state = {"boxes": [], "people": [], "is_threat": False}
    prev_time = time.time()
    fps = 0
    alarm_sounding = False
    while not stop_event_arg.is_set():
        ret, frame_orig = cap.read()
        if stop_event_arg.is_set():
            break
        if not ret:
            break
        orig_h, orig_w = frame_orig.shape[:2]
        scale = RESIZE_WIDTH / orig_w
        new_h = int(orig_h * scale)
        frame_resized = cv2.resize(frame_orig, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        try:
            frame_queue.put(frame_resized, block=False)
        except queue.Full:
            pass
        try:
            last_detection_state = result_queue.get(block=False)
        except queue.Empty:
            pass
        current_logs = []
        is_threat_detected = last_detection_state["is_threat"]
        
        # --- Log generation logic ---
        has_threat_log = False
        has_notice_log = False
        if "boxes" in last_detection_state:
            for (box, label, confidence, threat_level) in last_detection_state["boxes"]:
                if threat_level == 2:
                    current_logs.append(f"THREAT: {label.upper()} (ACTIVE)")
                    has_threat_log = True
                elif threat_level == 1:
                    current_logs.append(f"NOTICE: {label} (Carrying)")
                    has_notice_log = True
        
        fall_count = 0
        if "people" in last_detection_state:
            people_count = len(last_detection_state["people"])
            if people_count > 0:
                current_logs.append(f"PEOPLE: {people_count} detected")
                for (box, keypoints_xy, is_falling) in last_detection_state["people"]:
                    if is_falling:
                        fall_count += 1
                if fall_count > 0:
                    current_logs.append(f"THREAT: {fall_count} PERSON(S) FALLEN")
                    has_threat_log = True
        
        if not has_threat_log and not has_notice_log and fall_count == 0:
             current_logs.append("All clear.")
        # --- End log generation ---

        with log_lock:
            global_logs = current_logs
            
        if is_threat_detected and not alarm_sounding:
            alarm_sounding = True
            save_capture(frame_orig, current_logs)
            if sound_file_exists:
                try:
                    winsound.PlaySound(WARNING_SOUND_FILE, winsound.SND_ASYNC)
                except Exception as e:
                    print(f"Error playing sound: {e}")
        elif not is_threat_detected and alarm_sounding:
            alarm_sounding = False
            
        display_frame = draw_overlays(frame_orig, last_detection_state, scale=(1/scale))
        curr_time = time.time()
        if (curr_time - prev_time) > 0:
            fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_FPS, 2)
        with frame_lock:
            output_frame = display_frame.copy()
    print("Stopping video processing...")
    worker.join()
    cap.release()
    with frame_lock:
        output_frame = None
    with log_lock:
        global_logs = ["Scanning stopped."]


# --- 9. Flask Web Server ---
# (This section is unchanged)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scan")
def scan():
    return render_template("scan.html")

@app.route("/captures")
def captures():
    return render_template("captures.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/tech")
def tech():
    return render_template("tech.html")

@app.route("/tips")
def tips():
    return render_template("tips.html")

@app.route("/logs")
def get_logs():
    with log_lock:
        logs_copy = list(global_logs)
    return jsonify(logs=logs_copy)

@app.route("/api/captures")
def api_captures():
    with captures_lock:
        captures_copy = list(global_captures)
    return jsonify(captures=captures_copy)

@app.route("/api/clear_captures", methods=['POST'])
def api_clear_captures():
    global global_captures
    with captures_lock:
        global_captures = []
    return jsonify(success=True)

def stream_generator():
    global output_frame, frame_lock, scanning_active
    while True:
        with frame_lock:
            if not scanning_active or output_frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "SCANNING STOPPED", (110, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                (flag, encoded_image) = cv2.imencode(".jpg", placeholder)
            else:
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        time.sleep(0.01)

@app.route("/video_feed")
def video_feed():
    return Response(stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/start_scanning", methods=['POST'])
def api_start_scanning():
    global scanning_active, video_thread, stop_event
    with scanning_lock:
        if not scanning_active:
            scanning_active = True
            stop_event = threading.Event()
            video_thread = threading.Thread(target=start_video_processing, args=(stop_event,))
            video_thread.daemon = True
            video_thread.start()
            print("Scan started.")
            return jsonify(status="started")
    return jsonify(status="already_running")

@app.route("/api/stop_scanning", methods=['POST'])
def api_stop_scanning():
    global scanning_active, video_thread, stop_event
    with scanning_lock:
        if scanning_active:
            scanning_active = False
            if stop_event:
                stop_event.set()
            if video_thread:
                video_thread.join()
            video_thread = None
            stop_event = None
            print("Scan stopped.")
            return jsonify(status="stopped")
    return jsonify(status="already_stopped")

@app.route("/api/scan_status")
def api_scan_status():
    global scanning_active
    with scanning_lock:
        return jsonify(is_scanning=scanning_active)

# --- Main execution ---
if __name__ == "__main__":
    print("\nStarting web server... Open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=False)