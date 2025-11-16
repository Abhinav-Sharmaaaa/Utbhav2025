import cv2
import numpy as np
import time
import threading
import queue
from flask import Flask, render_template, Response, jsonify, redirect, url_for
import winsound
import os
from ultralytics import YOLO
import base64  # <-- ADDED
from datetime import datetime  # <-- ADDED

# --- 1. Flask & Threading Setup ---
output_frame = None
frame_lock = threading.Lock()
global_logs = []
log_lock = threading.Lock()
global_captures = []  # <-- ADDED
captures_lock = threading.Lock()  # <-- ADDED

# Configure Flask to use frontend folder
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(os.path.dirname(base_dir), 'frontend')

app = Flask(__name__, 
            template_folder=frontend_dir,
            static_folder=frontend_dir,
            static_url_path='')

# --- 2. Constants and Configuration ---
RESIZE_WIDTH = 640 

# --- YOLOv8 Config ---
MODEL_FILE_V8_POSE = "yolov8n-pose.pt" # For Fall Detection
MODEL_FILE_V8_OBJ = "yolov8n.pt"      # For Object Detection

CONF_THRESHOLD = 0.3 
NMS_THRESHOLD = 0.4 
DANGEROUS_OBJECTS = ["cell phone", "knife", "scissors", "baseball bat", "bottle"] 

# ### IMPROVED MOTION DETECTION THRESHOLDS ###
MOTION_THRESHOLD = 12.0  # Threshold for actual movement
MIN_ROI_SIZE = 30  # Minimum box size to track
MOTION_FRAME_COUNT = 2  # Number of frames with motion before triggering (reduced)
MOTION_HISTORY_SIZE = 8  # Keep longer history of motion values
MOTION_WINDOW_TIME = 1.5  # Seconds - look for motion within this time window

# Keypoints needed for fall detection
BODY_PARTS = {
    "L-shoulder": 5,
    "R-shoulder": 6,
    "L-hip": 11,
    "R-hip": 12
}

# --- Colors ---
COLOR_THREAT_ACTIVE = (0, 0, 255)    # Red (Active Threat)
COLOR_THREAT_NOTICE = (255, 255, 0)  # Cyan (Carrying Object)
COLOR_PERSON_SAFE = (0, 255, 0)      # Green (Person OK)
COLOR_FPS = (255, 255, 255)          # White

# --- Sound file configuration ---
WARNING_SOUND_FILE = "warning.wav"
sound_file_exists = os.path.exists(WARNING_SOUND_FILE)
if not sound_file_exists:
    print(f"Warning: Sound file '{WARNING_SOUND_FILE}' not found.")

# --- 3. Load Models ---
print("Loading models...")

# Load YOLOv8-Pose (for people/poses)
try:
    model_yolo_v8_pose = YOLO(MODEL_FILE_V8_POSE)
    print("YOLOv8-Pose loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8-Pose model: {e}")
    exit()

# Load YOLOv8-Object (for objects)
try:
    model_yolo_v8_obj = YOLO(MODEL_FILE_V8_OBJ)
    classes_v8_obj = model_yolo_v8_obj.names
    print("YOLOv8-Object loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8-Object model: {e}")
    print("Please make sure you have 'yolov8n.pt' in the same directory.")
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


# --- 5. Motion Tracker Class ---
class MotionTracker:
    """Track motion history for objects with time-based windowing"""
    def __init__(self):
        self.object_motion_history = {}  # {label: [(timestamp, motion_value)]}
        self.object_last_high_motion = {}  # {label: timestamp}
    
    def update(self, label, motion_magnitude, timestamp):
        """Update motion history and return threat level"""
        # Initialize if new object
        if label not in self.object_motion_history:
            self.object_motion_history[label] = []
            self.object_last_high_motion[label] = 0
        
        # Add to history with timestamp
        self.object_motion_history[label].append((timestamp, motion_magnitude))
        
        # Remove old entries outside time window
        cutoff_time = timestamp - MOTION_WINDOW_TIME
        self.object_motion_history[label] = [
            (t, m) for t, m in self.object_motion_history[label] 
            if t > cutoff_time
        ]
        
        # Keep maximum history size
        if len(self.object_motion_history[label]) > MOTION_HISTORY_SIZE:
            self.object_motion_history[label] = self.object_motion_history[label][-MOTION_HISTORY_SIZE:]
        
        # Count frames with significant motion in the time window
        motion_frames = sum(1 for _, m in self.object_motion_history[label] if m > MOTION_THRESHOLD)
        
        # Get recent motion values (last 3 frames)
        recent_motions = [m for _, m in self.object_motion_history[label][-3:]]
        max_recent_motion = max(recent_motions) if recent_motions else 0
        avg_recent_motion = np.mean(recent_motions) if recent_motions else 0
        
        # Update last high motion timestamp
        if motion_magnitude > MOTION_THRESHOLD:
            self.object_last_high_motion[label] = timestamp
        
        # Time since last high motion
        time_since_motion = timestamp - self.object_last_high_motion[label]
        
        # Threat detection logic with multiple criteria
        # Active threat if:
        # 1. Recent high motion (within 0.5 seconds)
        # 2. Multiple motion frames in window OR sustained high motion
        if time_since_motion < 0.5 and (
            motion_frames >= MOTION_FRAME_COUNT or 
            avg_recent_motion > MOTION_THRESHOLD * 0.8
        ):
            return 2  # Active threat
        
        # Notice if any motion detected recently
        elif max_recent_motion > MOTION_THRESHOLD * 0.6 or motion_frames >= 1:
            return 1  # Notice - some movement
        else:
            return 1  # Just carrying
    
    def cleanup_old_objects(self, current_labels):
        """Remove tracking for objects no longer detected"""
        existing_labels = set(self.object_motion_history.keys())
        for label in existing_labels - set(current_labels):
            del self.object_motion_history[label]
            del self.object_last_high_motion[label]


# --- 6. Worker Thread Function ---
def model_worker(frame_queue, result_queue, stop_event):
    global classes_v8_obj 
    prev_gray = None
    motion_tracker = MotionTracker()
    start_time = time.time()  # Track time for motion history
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1) 
        except queue.Empty:
            continue
        
        current_time = time.time() - start_time  # Relative timestamp
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Skip first frame
        if prev_gray is None:
            prev_gray = gray
            continue
        
        current_state = {"boxes": [], "people": [], "is_threat": False}
        is_threat = False
        detected_dangerous_objects = []

        # --- A: YOLOv8 Object Detection & Motion ---
        results_obj_list = model_yolo_v8_obj(frame, verbose=False, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)
        
        if results_obj_list:
            result_obj = results_obj_list[0] 
            
            if result_obj.boxes is not None:
                for i in range(len(result_obj.boxes)):
                    box_xywh = result_obj.boxes.xywh[i].cpu().numpy()
                    class_id = int(result_obj.boxes.cls[i].cpu().numpy())
                    label = classes_v8_obj[class_id]
                    
                    if label == "person":
                        continue
                        
                    confidence = float(result_obj.boxes.conf[i].cpu().numpy())
                    
                    x = int(box_xywh[0] - box_xywh[2] / 2)
                    y = int(box_xywh[1] - box_xywh[3] / 2)
                    w_box = int(box_xywh[2])
                    h_box = int(box_xywh[3])
                    
                    box_for_flow = [x, y, w_box, h_box]
                    threat_level = 0 

                    if label in DANGEROUS_OBJECTS:
                        detected_dangerous_objects.append(label)
                        
                        # Check if box is too small (too noisy)
                        if w_box < MIN_ROI_SIZE or h_box < MIN_ROI_SIZE:
                            threat_level = 1  # Too small to track reliably
                        else:
                            # Ensure coordinates are within bounds
                            x, y = max(0, x), max(0, y)
                            x_max, y_max = min(x + w_box, frame.shape[1]), min(y + h_box, frame.shape[0])
                            
                            roi_prev = prev_gray[y:y_max, x:x_max]
                            roi_curr = gray[y:y_max, x:x_max]
                            
                            # Check if ROIs are valid and same shape
                            if (roi_prev.shape[0] > MIN_ROI_SIZE and 
                                roi_prev.shape[1] > MIN_ROI_SIZE and 
                                roi_prev.shape == roi_curr.shape):
                                
                                # Strong blur to reduce noise
                                roi_prev_blur = cv2.GaussianBlur(roi_prev, (9, 9), 0)
                                roi_curr_blur = cv2.GaussianBlur(roi_curr, (9, 9), 0)
                                
                                # Calculate optical flow
                                flow = cv2.calcOpticalFlowFarneback(
                                    roi_prev_blur, roi_curr_blur, None, 
                                    0.5, 3, 15, 3, 5, 1.2, 0
                                )
                                
                                # Calculate magnitude
                                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                
                                # Use 90th percentile to ignore noise
                                motion_magnitude = np.percentile(mag, 90)
                                
                                # Use motion tracker for temporal filtering with timestamp
                                threat_level = motion_tracker.update(label, motion_magnitude, current_time)
                            else:
                                threat_level = 1  # Invalid ROI, just notice
                            
                        if threat_level == 2:
                            is_threat = True
                        
                        current_state["boxes"].append((box_for_flow, label, confidence, threat_level))
        
        # Clean up tracking for objects no longer detected
        motion_tracker.cleanup_old_objects(detected_dangerous_objects)

        # --- B: YOLOv8 Pose Detection ---
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
        
        prev_gray = gray


# --- 7. Drawing Function ---
def draw_overlays(frame, state, scale):
    for (box, label, confidence, threat_level) in state["boxes"]:
        x, y, w, h = [int(v * scale) for v in box]
        if threat_level == 2: 
            color = COLOR_THREAT_ACTIVE
            text = f"THREAT: {label.upper()} (ACTIVE)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif threat_level == 1: 
            color = COLOR_THREAT_NOTICE
            text = f"NOTICE: {label} (Carrying)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

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

# --- [NEW] Function to save captures ---
def save_capture(frame, logs):
    global global_captures, captures_lock
    
    # Get the first (most severe) threat log as description
    description = "Threat Detected"
    for log in logs:
        if log.startswith("THREAT"):
            description = log
            break
            
    # Encode image to JPEG and then to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    capture_data = {
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "image": img_base64
    }
    
    with captures_lock:
        global_captures.insert(0, capture_data) # Add to start of list
        # Optional: Limit number of captures
        # global_captures = global_captures[:50] 

# --- 8. Main Video Loop (Threaded Function) ---
def start_video_processing():
    global output_frame, frame_lock, global_logs, log_lock, sound_file_exists
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = threading.Thread(target=model_worker, args=(frame_queue, result_queue, stop_event))
    worker.daemon = True
    worker.start()

    last_detection_state = {"boxes": [], "people": [], "is_threat": False}
    prev_time = time.time()
    fps = 0
    alarm_sounding = False

    while True:
        ret, frame_orig = cap.read()
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
        
        for (box, label, confidence, threat_level) in last_detection_state["boxes"]:
            if threat_level == 2:
                current_logs.append(f"THREAT: {label.upper()} (ACTIVE)")
            elif threat_level == 1:
                current_logs.append(f"NOTICE: {label} (Carrying)")
        
        people_count = len(last_detection_state["people"])
        fall_count = 0
        if people_count > 0:
            current_logs.append(f"PEOPLE: {people_count} detected")
            for (box, keypoints_xy, is_falling) in last_detection_state["people"]:
                if is_falling:
                    fall_count += 1
            if fall_count > 0:
                current_logs.append(f"THREAT: {fall_count} PERSON(S) FALLEN")

        if not current_logs:
             current_logs.append("All clear.")
        
        with log_lock:
            global_logs = current_logs
        
        if is_threat_detected and not alarm_sounding:
            alarm_sounding = True
            
            # --- MODIFIED: Save capture on new threat ---
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
    stop_event.set()
    worker.join()
    cap.release()


# --- 9. Flask Web Server ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scan")
def scan():
    return render_template("scan.html")

# --- [NEW] Route for captures page ---
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

# --- [NEW] API endpoint for getting captures ---
@app.route("/api/captures")
def api_captures():
    with captures_lock:
        captures_copy = list(global_captures)
    return jsonify(captures=captures_copy)

# --- [NEW] API endpoint for clearing captures ---
@app.route("/api/clear_captures", methods=['POST'])
def api_clear_captures():
    global global_captures
    with captures_lock:
        global_captures = []
    return jsonify(success=True)

def stream_generator():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                continue
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


# --- Main execution ---
if __name__ == "__main__":
    video_thread = threading.Thread(target=start_video_processing)
    video_thread.daemon = True
    video_thread.start()
    
    print("\nStarting web server... Open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=False)