import cv2
import time
import threading
from queue import Queue

class BaseDetector:
    def __init__(self, cctv_name, url, app=None, socketio=None, db=None, ResultModel=None):
        self.cctv_name = cctv_name
        self.url = url
        self.app = app
        self.socketio = socketio
        self.db = db
        self.ResultModel = ResultModel
        
        self.is_running = True
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.alert_queue = Queue()  # 모든 자식이 공통으로 사용할 비동기 큐
        
        # 🛡️ 비동기 알람 워커 실행 (부모가 관리)
        self.start_alert_worker()

    def start_alert_worker(self):
        """DB 저장 및 소켓 전송만 전담하는 별도 스레드"""
        def worker():
            while self.is_running:
                if not self.alert_queue.empty():
                    # 자식 클래스에서 구현한 process_alert을 호출 (다형성)
                    data = self.alert_queue.get()
                    try:
                        self.process_alert(data)
                    except Exception as e:
                        print(f"❌ [Worker Error] {self.cctv_name}: {e}")
                time.sleep(0.1)
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def generate_frames(self):
        last_frame = None
        
        while self.is_running:

            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:

                if last_frame is None or last_frame is not frame:

                    last_frame = frame

                    ret, buffer = cv2.imencode(
                        ".jpg",
                        last_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 60]
                    )

                    if not ret:
                        continue

                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        buffer.tobytes() +
                        b'\r\n'
                    )

            time.sleep(0.03)        # 실제 CCTV는 10~15fps >> 1/0.03

    def process_alert(self, data):
        """자식 클래스(Fire, Reverse)에서 각각 다르게 구현할 부분"""
        raise NotImplementedError("자식 클래스에서 process_alert를 반드시 구현해야 합니다.")

    def stop(self):
        """자원 해제 공통 메서드"""
        self.is_running = False
        print(f"🛑 [{self.cctv_name}] 분석 프로세스 종료")