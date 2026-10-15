import math
import os
import random
import struct
import threading
import time
from pathlib import Path
from flask import Flask, jsonify, send_file, request
import serial

BASE = Path(__file__).parent
SIMULATE = os.getenv("SIMULATE", "0") == "1"
SERIAL_PORT = os.getenv("LDS_PORT", "/dev/ttyUSB0")
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "/dev/ttyACM0")
BAUD = 115200
PORT = int(os.getenv("PORT", "5002"))

PKT_LEN = 47
DIST_MIN_MM, DIST_MAX_MM = 160, 8000
STALE_SEC = 1.0

lock = threading.Lock()
dist_mm = [0] * 360
persistent_map = [0] * 360
conf = [0] * 360
stamp = [0.0] * 360
stats = {
    "connected": False, "rpm": 0.0, "packets": 0, "crc_ok": 0, "crc_bad": 0,
    "crc_checked": True, "last_packet": 0.0,
}
sensor_data = {"t": None, "h": None, "mq2": None, "mq135": None, "updated": 0.0}

def arduino_loop():
    while True:
        try:
            with serial.Serial(ARDUINO_PORT, 9600, timeout=2.0) as ser:
                print(f"[arduino] opened {ARDUINO_PORT}")
                while True:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("{") and line.endswith("}"):
                        import json
                        try:
                            data = json.loads(line)
                            with lock:
                                sensor_data["t"] = data.get("t")
                                sensor_data["h"] = data.get("h")
                                sensor_data["mq2"] = data.get("mq2")
                                sensor_data["mq135"] = data.get("mq135")
                                sensor_data["updated"] = time.time()
                        except:
                            pass
        except Exception as e:
            print(f"[arduino] {e} - retrying in 3 seconds")
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
    now = time.time()
    with lock:
        for i in range(12):
            d, c = struct.unpack_from("<HB", pkt, 6 + 3 * i)
            idx = to_robot_index(start_deg + step * i)
            if DIST_MIN_MM <= d <= DIST_MAX_MM and c >= 80:
                dist_mm[idx] = d
                conf[idx] = c
                persistent_map[idx] = d
            else:
                dist_mm[idx] = 0
                conf[idx] = 0
            stamp[idx] = now
        rpm = speed / 6.0
        stats["rpm"] = rpm if stats["rpm"] == 0 else stats["rpm"] * 0.95 + rpm * 0.05
        stats["packets"] += 1
        stats["last_packet"] = now
        stats["connected"] = True

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
            if stats["crc_checked"]:
                if crc8(pkt[:-1]) == pkt[-1]:
                    stats["crc_ok"] += 1
                else:
                    stats["crc_bad"] += 1
                    if stats["crc_ok"] == 0 and stats["crc_bad"] >= 300:
                        stats["crc_checked"] = False
                    del self.buf[:i + 1]
                    continue
            handle_packet(pkt)
            del self.buf[:i + PKT_LEN]

def serial_loop():
    parser = PacketParser()
    while True:
        try:
            with serial.Serial(SERIAL_PORT, BAUD, timeout=1) as port:
                print(f"[lidar] opened {SERIAL_PORT}")
                while True:
                    data = port.read(256)
                    if data:
                        parser.feed(data)
        except Exception as e:
            stats["connected"] = False
            print(f"[lidar] {e} - retrying in 3 seconds")
            time.sleep(3)

app = Flask(__name__)

@app.after_request
def add_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/")
def index():
    return send_file(BASE / "lidar.html")

@app.route("/api/scan")
def api_scan():
    now = time.time()
    with lock:
        ranges = list(persistent_map)
    return jsonify({"ts": now, "ranges_mm": ranges})

@app.route("/api/clear_map", methods=["POST"])
def api_clear_map():
    global persistent_map
    with lock:
        persistent_map = [0] * 360
    return jsonify({"status": "ok"})

@app.route("/api/status")
def api_status():
    now = time.time()
    with lock:
        out = dict(stats)
        valid = sum(1 for i in range(360) if persistent_map[i] > 0)
        sensors = dict(sensor_data)
    out["connected"] = bool(out["connected"] and now - out["last_packet"] < 2.0)
    out["rpm"] = round(out["rpm"], 1)
    out["valid_points"] = valid
    out["simulate"] = SIMULATE
    out["port"] = SERIAL_PORT
    out["sensors"] = sensors
    return jsonify(out)

if __name__ == "__main__":
    threading.Thread(target=serial_loop, daemon=True).start()
    threading.Thread(target=arduino_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
