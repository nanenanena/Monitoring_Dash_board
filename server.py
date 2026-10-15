"""Raspberry Pi 4B disaster robot integrated dashboard server for Rescue Workers.

- Lidar LDS-02 2D SLAM Mapping via BreezySLAM
- Arduino sensor readings & 48-hour history (MQ-2, MQ-135, Temp, Hum)
- NoIR camera stream & Dual USB Webcams with MobileNet-SSD Fall Detection (Debounced)
- System Health (CPU Temp, Wi-Fi RSSI, Battery)
- SLAM Map Hazard & Victim Pinning Markers
- Incident Event Logbook & Recent Logs for Main Dashboard Feed
"""
import cv2
import glob
import io
import json
import math
import os
import struct
import threading
import time
from collections import deque
from threading import Condition

import numpy as np
import serial
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from flask import Flask, Response, jsonify, request, send_from_directory

BAUD_ARDUINO = 9600
BAUD_LIDAR = 115200
SERIAL_PORT_ARDUINO = os.environ.get('SERIAL_PORT')
LDS_PORT = os.environ.get('LDS_PORT', '/dev/ttyUSB0')

FIELDS = ('mq2', 'mq135', 'temp', 'hum')
UNITS = {
    'second': (0, '%H:%M:%S'),
    'minute': (60, '%H:%M'),
    'hour': (3600, '%d %H:00'),
}

app = Flask(__name__, static_folder='static', static_url_path='/static')
lock = threading.Lock()
samples = deque(maxlen=86400)
latest = None
last_ts = None
port_name = None

PKT_LEN = 47
persistent_map = [0] * 360
lidar_stats = {
    "connected": False, "rpm": 0.0, "packets": 0, "crc_ok": 0, "crc_bad": 0,
    "crc_checked": True, "last_packet": 0.0,
}

laser = Laser(360, 5, 360, 0)
MAP_SIZE_PIXELS = 400
MAP_SIZE_METERS = 20
slam = RMHC_SLAM(laser, map_size_pixels=MAP_SIZE_PIXELS, map_size_meters=MAP_SIZE_METERS)
map_bytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)

os.makedirs('static', exist_ok=True)
fall_status = {"detected": False, "time": "", "image": ""}
markers = []
incident_logs = []

# Fall detection debounce variables
fall_consecutive_count = 0
last_fall_alert_time = 0

net = None
try:
    if os.path.exists('deploy.prototxt') and os.path.exists('mobilenet_iter_73000.caffemodel'):
        net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
        print("[dnn] MobileNet-SSD model loaded successfully")
    else:
        print("[dnn] Warning: MobileNet-SSD files not found.")
except Exception as e:
    print("[dnn] Failed to load model:", e)

def get_system_health():
    cpu_temp = 45.0
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = round(float(f.read().strip()) / 1000.0, 1)
    except:
        pass

    wifi_rssi = -65
    try:
        with open("/proc/net/wireless", "r") as f:
            lines = f.readlines()
            if len(lines) > 2:
                parts = lines[2].split()
                wifi_rssi = int(float(parts[3]))
    except:
        pass

    return {"cpu_temp": cpu_temp, "wifi_rssi": wifi_rssi, "battery": 88}

def num(text):
    try:
        v = float(text)
        return None if math.isnan(v) else v
    except ValueError:
        return None

def find_arduino_port():
    if SERIAL_PORT_ARDUINO:
        return SERIAL_PORT_ARDUINO
    ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    return ports[0] if ports else None

def arduino_loop():
    global latest, last_ts, port_name
    while True:
        port = find_arduino_port()
        if not port:
            time.sleep(3)
            continue
        try:
            with serial.Serial(port, BAUD_ARDUINO, timeout=3) as ser:
                with lock:
                    port_name = port
                time.sleep(2)
                ser.reset_input_buffer()
                while True:
                    line = ser.readline().decode('utf-8', 'ignore').strip()
                    if not line:
                        continue
                    vals = [None, None, None, None]
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            vals = [data.get('mq2'), data.get('mq135'), data.get('t'), data.get('h')]
                        except:
                            continue
                    else:
                        parts = line.split(',')
                        if len(parts) == 5 and parts[0] == 'DATA':
                            vals = [num(p) for p in parts[1:]]
                        else:
                            continue
                    now = time.time()
                    with lock:
                        samples.append((now, *vals))
                        latest = dict(zip(FIELDS, vals))
                        last_ts = now
                        
                        if (vals[0] and vals[0] > 600) or (vals[1] and vals[1] > 600):
                            hazard_name = "MQ-2 Gas Danger" if (vals[0] and vals[0] > 600) else "MQ-135 Air Danger"
                            if not incident_logs or time.time() - incident_logs[0].get('timestamp', 0) > 30:
                                incident_logs.insert(0, {
                                    "timestamp": now,
                                    "time": time.strftime('%H:%M:%S', time.localtime(now)),
                                    "type": hazard_name,
                                    "image": "",
                                    "values": dict(zip(FIELDS, vals))
                                })
        except (serial.SerialException, OSError) as e:
            print('Arduino serial error:', e)
            with lock:
                port_name = None
            time.sleep(3)

def _make_crc_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x4D) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        table.append(c)
    return table

CRC_TABLE = _make_crc_table()

def crc8(data):
    c = 0
    for b in data:
        c = CRC_TABLE[(c ^ b) & 0xFF]
    return c

def to_robot_index(raw_deg):
    return int(round(raw_deg)) % 360

def handle_packet(pkt):
    speed, start = struct.unpack_from("<HH", pkt, 2)
    end, _ts = struct.unpack_from("<HH", pkt, 42)
    start_deg = start / 100.0
    end_deg = end / 100.0
    if end_deg < start_deg:
        end_deg += 360.0
    step = (end_deg - start_deg) / 11.0
    
    current_scan = [0] * 360
    with lock:
        for i in range(12):
            d, c = struct.unpack_from("<HB", pkt, 6 + 3 * i)
            idx = to_robot_index(start_deg + step * i)
            if 200 <= d <= 8000 and c >= 100:
                persistent_map[idx] = d
                current_scan[idx] = d
            
        try:
            slam.update(current_scan)
            slam.getmap(map_bytes)
        except Exception as e:
            print("SLAM update error:", e)

        rpm = speed / 6.0
        lidar_stats["rpm"] = rpm if lidar_stats["rpm"] == 0 else lidar_stats["rpm"] * 0.95 + rpm * 0.05
        lidar_stats["packets"] += 1
        lidar_stats["last_packet"] = time.time()
        lidar_stats["connected"] = True

class PacketParser:
    def __init__(self):
        self.buf = bytearray()
    def feed(self, data):
        self.buf.extend(data)
        while True:
            i = self.buf.find(b"\x54\x2c")
            if i < 0:
                del self.buf[:-1]
                return
            if len(self.buf) - i < PKT_LEN:
                del self.buf[:i]
                return
            pkt = bytes(self.buf[i:i + PKT_LEN])
            if lidar_stats["crc_checked"]:
                if crc8(pkt[:-1]) == pkt[-1]:
                    lidar_stats["crc_ok"] += 1
                else:
                    lidar_stats["crc_bad"] += 1
                    if lidar_stats["crc_ok"] == 0 and lidar_stats["crc_bad"] >= 300:
                        lidar_stats["crc_checked"] = False
                    del self.buf[:i + 1]
                    continue
            handle_packet(pkt)
            del self.buf[:i + PKT_LEN]

def lidar_loop():
    parser = PacketParser()
    while True:
        try:
            with serial.Serial(LDS_PORT, BAUD_LIDAR, timeout=1) as port:
                print(f"[lidar] opened {LDS_PORT}")
                while True:
                    data = port.read(256)
                    if data:
                        parser.feed(data)
        except Exception as e:
            lidar_stats["connected"] = False
            print(f"[lidar] {e} - retrying in 3 seconds")
            time.sleep(3)

def avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None

def make_row(ts, fmt, values):
    row = {
        't': time.strftime(fmt, time.localtime(ts)),
        'full': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
    }
    row.update(zip(FIELDS, values))
    return row

def history(unit, n):
    step, fmt = UNITS[unit]
    cutoff = time.time() - n * step if step else 0
    rows = []
    with lock:
        for s in reversed(samples):
            if (step == 0 and len(rows) >= n) or s[0] < cutoff:
                break
            rows.append(s)
    rows.reverse()
    if step == 0:
        return [make_row(s[0], fmt, s[1:]) for s in rows]
    buckets = {}
    for s in rows:
        buckets.setdefault(int(s[0] // step), []).append(s)
    return [
        make_row(k * step, fmt, [avg([g[i] for g in buckets[k]]) for i in range(1, 5)])
        for k in sorted(buckets)
    ]

def query_args(default_n):
    unit = request.args.get('unit', 'second')
    if unit not in UNITS:
        unit = 'second'
    n = max(1, min(request.args.get('n', default_n, type=int), 5000))
    return unit, n

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()

noir_output = StreamingOutput()
camera_ok = False
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={'size': (640, 480)}, controls={'FrameRate': 15}))
    picam2.start_recording(MJPEGEncoder(), FileOutput(noir_output))
    camera_ok = True
except Exception as e:
    print('NoIR Camera unavailable:', e)

def gen_noir_frames():
    while True:
        with noir_output.condition:
            if not noir_output.condition.wait(timeout=5):
                continue
            frame = noir_output.frame
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

def gen_usb_webcam_frames(preferred_indices):
    global fall_status, fall_consecutive_count, last_fall_alert_time
    cap = None
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"[webcam] successfully opened device {idx}")
            break
    
    if not cap or not cap.isOpened():
        print("[webcam] warning: no webcam found")
        while True:
            time.sleep(1)
            yield b''

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    frame_count = 0
    last_boxes = []

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        if frame is not None:
            frame = frame[:, :-2]

        frame_count += 1
        if net is not None and frame_count % 15 == 0:
            try:
                (h, w) = frame.shape[:2]
                small_frame = cv2.resize(frame, (160, 120))
                blob = cv2.dnn.blobFromImage(small_frame, 0.007843, (160, 120), 127.5)
                net.setInput(blob)
                detections = net.forward()
                
                last_boxes = []
                fallen_detected_current = False

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5 and int(detections[0, 0, i, 1]) == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        box_w = endX - startX
                        box_h = endY - startY
                        is_fallen = (box_w > box_h * 1.2)
                        
                        last_boxes.append((startX, startY, endX, endY, is_fallen))
                        if is_fallen:
                            fallen_detected_current = True

                with lock:
                    if fallen_detected_current:
                        fall_consecutive_count += 1
                    else:
                        fall_consecutive_count = max(0, fall_consecutive_count - 1)

                    now_t = time.time()
                    if fall_consecutive_count >= 3 and not fall_status["detected"] and (now_t - last_fall_alert_time > 30):
                        capture_path = "static/fall_capture.jpg"
                        cv2.imwrite(capture_path, frame)
                        fall_status["detected"] = True
                        fall_status["time"] = time.strftime('%H:%M:%S')
                        fall_status["image"] = "/" + capture_path + "?t=" + str(now_t)
                        last_fall_alert_time = now_t
                        
                        incident_logs.insert(0, {
                            "timestamp": now_t,
                            "time": time.strftime('%H:%M:%S'),
                            "type": "Fallen Person Detected",
                            "image": "/" + capture_path,
                            "values": dict(zip(FIELDS, [latest.get(k) for k in FIELDS])) if latest else {}
                        })
            except Exception as e:
                pass

        for (startX, startY, endX, endY, is_fallen) in last_boxes:
            color = (0, 0, 255) if is_fallen else (0, 255, 0)
            label = "FALLEN!" if is_fallen else "Person"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, max(startY - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    cap.release()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    for folder in (os.path.join(BASE_DIR, 'static'), BASE_DIR):
        if os.path.isfile(os.path.join(folder, 'index.html')):
            return send_from_directory(folder, 'index.html')
    return 'index.html not found.', 404

@app.route('/video_feed/noir')
def video_feed_noir():
    if not camera_ok:
        return 'NoIR Camera unavailable', 503
    return Response(gen_noir_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/front')
def video_feed_front():
    return Response(gen_usb_webcam_frames([0, 2, 4]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/back')
def video_feed_back():
    return Response(gen_usb_webcam_frames([1, 3, 2]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    unit, n = query_args(20)
    with lock:
        cur = dict(latest) if latest else None
        age = time.time() - last_ts if last_ts else None
        port = port_name
        fall = dict(fall_status)
        health = get_system_health()
        recent_logs = list(incident_logs[:3])
    return jsonify(
        status={'arduino': age is not None and age < 6, 'port': port, 'camera': camera_ok},
        latest=cur,
        history=history(unit, n),
        fall=fall,
        health=health,
        logs=recent_logs,
        time=time.strftime('%Y-%m-%d %H:%M:%S'),
    )

@app.route('/api/scan')
def api_scan():
    now = time.time()
    with lock:
        ranges = list(persistent_map)
    return jsonify({"ts": now, "ranges_mm": ranges})

@app.route('/api/slam_image')
def api_slam_image():
    with lock:
        img_array = np.frombuffer(map_bytes, dtype=np.uint8).reshape((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
    success, encoded_image = cv2.imencode('.png', img_array)
    if not success:
        return "Image encode failed", 500
    return Response(encoded_image.tobytes(), mimetype='image/png')

@app.route('/api/markers', methods=['GET', 'POST', 'DELETE'])
def api_markers():
    global markers
    if request.method == 'GET':
        with lock:
            return jsonify(markers)
    elif request.method == 'POST':
        data = request.json
        with lock:
            markers.append(data)
        return jsonify({"status": "ok"})
    elif request.method == 'DELETE':
        with lock:
            markers.clear()
        return jsonify({"status": "ok"})

@app.route('/api/logs')
def api_logs():
    with lock:
        return jsonify(incident_logs)

@app.route('/api/clear_map', methods=["POST"])
def api_clear_map():
    global persistent_map, slam
    with lock:
        persistent_map = [0] * 360
        slam = RMHC_SLAM(laser, map_size_pixels=MAP_SIZE_PIXELS, map_size_meters=MAP_SIZE_METERS)
    return jsonify({"status": "ok"})

@app.route('/api/clear_fall', methods=["POST"])
def api_clear_fall():
    global fall_status, fall_consecutive_count
    with lock:
        fall_status["detected"] = False
        fall_consecutive_count = 0
    return jsonify({"status": "ok"})

@app.route('/api/export.csv')
def export_csv():
    unit, n = query_args(1800)
    lines = ['Time,MQ2_raw,MQ135_raw,Temperature_C,Humidity_RH']
    for r in history(unit, n):
        cells = [r['full']] + ['' if r[k] is None else str(r[k]) for k in FIELDS]
        lines.append(','.join(cells))
    return Response(
        '\n'.join(lines) + '\n', mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sensor_%s.csv' % unit})

if __name__ == '__main__':
    threading.Thread(target=arduino_loop, daemon=True).start()
    threading.Thread(target=lidar_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
