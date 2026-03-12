import os

# 공유 상태 변수
current_broadcast_type = None
alert_sent_session = {"fire": False, "reverse": False, "webcam": False}
sim_coords = {"lat": 37.5665, "lng": 126.9780}
current_video_file = {"fire": None, "reverse": None}
latest_frames = {}

# 경로 설정
CAPTURE_DIR = os.path.join(os.getcwd(), "static", "captures")

ANOMALY_DATA = {
    "fire": {"type": "화재 발생"},
    "reverse": {"type": "역주행"},
    "webcam": {"type": "실시간 현장"}
}