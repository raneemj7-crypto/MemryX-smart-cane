#!/usr/bin/env python3
import argparse
import time
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Optional Torch
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional GUI
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except Exception:
    print("[INFO] GUI not available (tkinter/PIL not installed). Running in console mode.")

# ============================================================================
# CONFIG
# ============================================================================
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PEOPLE = {"person"}
OTHER_OBJECTS = {
    "dog", "cat", "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "cup", "chair", "couch", "bed", "dining table",
    "laptop", "cell phone", "book"
}
ALL_OBJECTS = VEHICLES | PEOPLE | OTHER_OBJECTS


# ============================================================================
# CAMERA DETECTION
# ============================================================================
def check_cameras(max_index=5):
    """Return first working camera index (or None)."""
    print("🔍 Checking available cameras...\n")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                print(f"[✅] Camera index {i} is working")
                return i
    print("🚫 No working camera found.")
    return None


# ============================================================================
# DISTANCE ESTIMATOR (stable + smoothed)
# ============================================================================
# ============================================================================
# FIXED DISTANCE ESTIMATOR (guaranteed alert for near objects)
# ============================================================================
class DistanceEstimator:
    """
    Improved distance estimator that forces WARNING/DANGER
    when object occupies large part of camera frame.
    Works even on webcams with inconsistent scaling.
    """
    def __init__(self, ema_alpha=0.4, default_k=900.0):
        self.alpha = float(ema_alpha)
        self.k = float(default_k)
        self._ema = {}

    def _ema_update(self, key, value):
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
        else:
            self._ema[key] = self.alpha * value + (1 - self.alpha) * prev
        return self._ema[key]

    def calculate_distance(self, bbox_height_pixels, object_class):
        """Estimate distance (in meters) based on bounding box height."""
        if bbox_height_pixels <= 0:
            return None

        # Get raw estimate (inverse proportional)
        raw_distance = self.k / bbox_height_pixels
        raw_distance = max(0.05, min(raw_distance, 10.0))
        smoothed = self._ema_update(object_class, raw_distance)

        # --- Force danger if object fills too much of frame ---
        # Assuming 480p camera height
        frame_height = 480
        ratio = bbox_height_pixels / frame_height

        if ratio > 0.6:       # >60% of frame → very close
            smoothed = 0.3
        elif ratio > 0.4:     # >40% → close
            smoothed = 0.7

        return smoothed

    @staticmethod
    def get_alert_level(distance):
        """Return SAFE/WARNING/DANGER based on thresholds."""
        if distance is None:
            return "UNKNOWN"
        if distance <= 0.5:
            return "DANGER"
        elif distance <= 1.0:
            return "WARNING"
        else:
            return "SAFE"



# ============================================================================
# OBJECT TRACKER
# ============================================================================
class ObjectTracker:
    def __init__(self, timeout=3.0):
        self.timeout = timeout
        self.last_seen = defaultdict(float)
        self.last_count = defaultdict(int)

    def should_announce(self, key, count=1):
        now = time.time()
        last_time = self.last_seen.get(key, 0)
        last_cnt = self.last_count.get(key, 0)
        if (now - last_time > self.timeout) or (count != last_cnt):
            self.last_seen[key] = now
            self.last_count[key] = count
            return True
        return False


# ============================================================================
# TRAFFIC LIGHT DETECTOR (robust + smoothed + inverted logic)
# ============================================================================
class TrafficLightDetector:
    def __init__(self, k_smooth=5, min_frames_confirm=3):
        self.history = []
        self.k = int(max(1, k_smooth))
        self.min_frames_confirm = int(max(1, min_frames_confirm))

    def _mask_counts(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        green = cv2.inRange(hsv, (38, 50, 50), (95, 255, 255))
        return red, green

    def _vote_vertical(self, roi_bgr):
        h, w = roi_bgr.shape[:2]
        if h < 20 or w < 20:
            return "UNKNOWN"

        roi_norm = cv2.resize(roi_bgr, (64, 96))
        red_mask, green_mask = self._mask_counts(roi_norm)
        H = roi_norm.shape[0]
        top = slice(0, H // 3)
        bot = slice(2 * H // 3, H)

        red_top = int(cv2.countNonZero(red_mask[top, :]))
        green_bot = int(cv2.countNonZero(green_mask[bot, :]))

        if red_top > green_bot * 1.3 and red_top > 40:
            return "RED"
        if green_bot > red_top * 1.3 and green_bot > 40:
            return "GREEN"
        return "UNKNOWN"

    def classify_from_yolo_roi(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "UNKNOWN", 0.0

        state = self._vote_vertical(roi)
        self.history.append(state)
        if len(self.history) > self.k:
            self.history.pop(0)
        vals, counts = np.unique(self.history, return_counts=True)
        voted = vals[np.argmax(counts)]
        conf = min(1.0, counts.max() / self.min_frames_confirm)
        if counts.max() < self.min_frames_confirm:
            return "NONE", conf
        return voted, conf


# ============================================================================
# GUI
# ============================================================================
class VisionGuideGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VisionGuide - Smart Safety Cane")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        self._build()

    def _build(self):
        title = tk.Label(self.root, text="VisionGuide - Smart Safety Cane",
                         font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#ffffff")
        title.pack(pady=10)

        main = tk.Frame(self.root, bg="#1a1a1a")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left = tk.Frame(main, bg="#2a2a2a", relief=tk.RAISED, borderwidth=2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(left, text="Camera Feed", font=("Arial", 14, "bold"),
                 bg="#2a2a2a", fg="#ffffff").pack(pady=5)
        self.video_canvas = tk.Label(left, bg="#000000")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        right = tk.Frame(main, bg="#2a2a2a", relief=tk.RAISED, borderwidth=2)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, ipadx=20)

        def section(parent, title):
            tk.Label(parent, text=title, font=("Arial", 12, "bold"),
                     bg="#2a2a2a", fg="#00aaff").pack(pady=(15, 5))

        tk.Label(right, text="Detection Results", font=("Arial", 14, "bold"),
                 bg="#2a2a2a", fg="#ffffff").pack(pady=10)

        section(right, "Traffic Light Status")
        self.traffic_light_label = tk.Label(right, text="NONE", font=("Arial", 32, "bold"),
                                            bg="#2a2a2a", fg="#888888", width=15, height=2,
                                            relief=tk.SUNKEN, borderwidth=3)
        self.traffic_light_label.pack(pady=10)

        section(right, "Distance Alert")
        self.distance_label = tk.Label(right, text="SAFE", font=("Arial", 20, "bold"),
                                       bg="#2a2a2a", fg="#00ff00", width=15, height=1,
                                       relief=tk.SUNKEN, borderwidth=3)
        self.distance_label.pack(pady=10)

        section(right, "Vehicles Detected")
        self.vehicles_text = tk.Text(right, font=("Arial", 12), bg="#1a1a1a", fg="#ffffff",
                                     height=4, width=25, relief=tk.SUNKEN, borderwidth=2)
        self.vehicles_text.pack(pady=5)

        section(right, "People Detected")
        self.people_label = tk.Label(right, text="0", font=("Arial", 24, "bold"),
                                     bg="#2a2a2a", fg="#ffffff", width=15, height=1,
                                     relief=tk.SUNKEN, borderwidth=3)
        self.people_label.pack(pady=10)

        section(right, "Other Objects")
        self.objects_text = tk.Text(right, font=("Arial", 12), bg="#1a1a1a", fg="#ffffff",
                                    height=4, width=25, relief=tk.SUNKEN, borderwidth=2)
        self.objects_text.pack(pady=5)

        self.fps_label = tk.Label(right, text="FPS: 0", font=("Arial", 10),
                                  bg="#2a2a2a", fg="#888888")
        self.fps_label.pack(pady=10)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 480))
        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.imgtk = imgtk
        self.video_canvas.configure(image=imgtk)

    def update_detections(self, det):
        tl = det.get("traffic_light", "NONE")
        # flip logic: RED=SAFE, GREEN=NOT SAFE
        if tl == "RED":
            self.traffic_light_label.config(text="SAFE", bg="#00ff00", fg="#000000")
        elif tl == "GREEN":
            self.traffic_light_label.config(text="NOT SAFE", bg="#ff0000", fg="#ffffff")
        else:
            self.traffic_light_label.config(text="NONE", bg="#2a2a2a", fg="#888888")

        alert = det.get("distance_alert", "SAFE")
        if alert == "DANGER":
            self.distance_label.config(text="DANGER", bg="#ff0000", fg="#ffffff")
        elif alert == "WARNING":
            self.distance_label.config(text="WARNING", bg="#ff8800", fg="#000000")
        else:
            self.distance_label.config(text="SAFE", bg="#2a2a2a", fg="#00ff00")

        vehicles = det.get("vehicles", [])
        self.vehicles_text.delete(1.0, tk.END)
        self.vehicles_text.insert(tk.END, "\n".join(vehicles) if vehicles else "None detected")

        self.people_label.config(text=str(det.get("people", 0)))

        objects = det.get("objects", [])
        self.objects_text.delete(1.0, tk.END)
        self.objects_text.insert(tk.END, "\n".join(objects) if objects else "None detected")

    def update_fps(self, fps):
        self.fps_label.config(text=f"FPS: {fps:.1f}")

    def run(self):
        self.root.mainloop()

    def destroy(self):
        try:
            self.root.quit()
        except Exception:
            pass


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    cam_index = check_cameras()
    if cam_index is None:
        return

    model = YOLO("yolov8n.pt")
    device = 0 if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    cap = cv2.VideoCapture(cam_index)
    distance_estimator = DistanceEstimator()
    tracker = ObjectTracker()
    traffic_light_detector = TrafficLightDetector()

    if GUI_AVAILABLE and not args.no_gui:
        gui = VisionGuideGUI()

        fps_start, fps_count = time.time(), 0
        def tick():
            nonlocal fps_start, fps_count
            ok, frame = cap.read()
            if not ok:
                gui.root.after(10, tick)
                return

            results = model.predict(frame, conf=0.4, verbose=False, device=device)[0]

            vehicles, objects, people = {}, set(), 0
            min_dist = float("inf")
            distance_alert = "SAFE"
            tl_state, tl_bbox = "NONE", None

            # Detect objects
            for box in results.boxes:
                name = results.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                bbox_h = y2 - y1
                dist = distance_estimator.calculate_distance(bbox_h, name)

                if dist and dist < min_dist:
                    min_dist = dist

                alert = DistanceEstimator.get_alert_level(dist)
                color = (0,255,0)
                if alert == "DANGER":
                    distance_alert = "DANGER"
                    color = (0,0,255)
                elif alert == "WARNING" and distance_alert != "DANGER":
                    distance_alert = "WARNING"
                    color = (0,165,255)

                if name in VEHICLES:
                    vehicles[name] = dist
                elif name in PEOPLE:
                    people += 1
                elif name in OTHER_OBJECTS:
                    objects.add(name)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=f"{name} {conf:.2f}"
                if dist: label+=f" {dist:.1f}m"
                cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

                if name=="traffic light":
                    tl_state, conf_tl = traffic_light_detector.classify_from_yolo_roi(frame,(x1,y1,x2,y2))
                    tl_bbox=(x1,y1,x2,y2)

            if tl_bbox:
                x1,y1,x2,y2=tl_bbox
                color=(0,0,255) if tl_state=="RED" else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                cv2.putText(frame,f"TL: {tl_state}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

            # Update GUI
            gui.update_frame(frame)
            gui.update_detections({
                "vehicles":[f"{n} - {d:.1f}m" for n,d in vehicles.items() if d],
                "people":people,
                "objects":list(sorted(objects)),
                "traffic_light":tl_state,
                "distance_alert":distance_alert
            })

            # FPS
            fps_count+=1
            if fps_count>=30:
                now=time.time()
                fps= fps_count/(now-fps_start)
                fps_start, fps_count = now, 0
                gui.update_fps(fps)
            gui.root.after(1,tick)

        print("✅ System ready. Press Ctrl+C to stop.")
        gui.root.after(1,tick)
        gui.run()
        cap.release()
        gui.destroy()
        return

    # Console-only fallback
    while True:
        ok, frame = cap.read()
        if not ok: break
        _ = model.predict(frame, conf=0.4, verbose=False, device=device)
        time.sleep(0.01)
    cap.release()

if __name__ == "__main__":
    main()
