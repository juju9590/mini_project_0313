import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
from .base_detector import BaseDetector 
from yolo_models.yolo_model import yolo_fire

class FireDetector(BaseDetector):
    def __init__(self, cctv_name, url, lat=37.5, lng=127.0, socketio=None, db=None, ResultModel=None, app=None):
        # 1. 부모 클래스(BaseDetector) 초기화 (비동기 큐 및 워커 자동 시작)
        super().__init__(cctv_name, url, app=app, socketio=socketio, db=db, ResultModel=ResultModel)
        
        self.lat = lat
        self.lng = lng
        
        # 2. 모델 로드 및 GPU 할당
        # imgsz=320 등으로 최적화하여 분석 속도 향상 가능
        # self.model = YOLO("best_SB.pt")
        self.model = yolo_fire

        # 3. FFMPEG 및 버퍼 설정 (지연 시간 최소화)
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 상태 제어 변수
        self.is_alerting = False 
        self.frame_count = 0
        self.frame_skip = 3  # 필요 시 프레임 스킵 (1이면 모든 프레임 읽음)

    def process_alert(self, data):
        """[Worker Thread] 부모의 큐에서 데이터를 받아 비동기로 DB/파일 저장"""
        frame, alert_time = data
        try:
            with self.app.app_context():
                # 1️⃣ 공통 결과 테이블 저장 (DetectionResult)
                new_alert = self.ResultModel(
                    event_type="fire",
                    address=self.cctv_name,
                    latitude=self.lat, 
                    longitude=self.lng,
                    detected_at=alert_time,
                    is_simulation=False,
                    is_resolved=False
                )
                self.db.session.add(new_alert)
                self.db.session.flush() # ID 확보용

                # 2️⃣ 이미지 파일 저장
                ts = alert_time.strftime("%Y%m%d_%H%M%S")
                filename = f"fire_real_{new_alert.id}_{ts}.jpg"
                save_path = os.path.join(self.app.root_path, "static", "captures")
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, filename)
                cv2.imwrite(filepath, frame)

                # 3️⃣ 화재 상세 테이블 저장 (FireResult)
                from models import FireResult 
                fire_detail = FireResult(
                    result_id=new_alert.id,
                    image_path=f"/static/captures/{filename}",
                    fire_severity="중간"  # 기본값 설정
                )
                self.db.session.add(fire_detail)
                self.db.session.commit()

                # 4️⃣ 소켓 알림 전송
                if self.socketio:
                    self.socketio.emit('anomaly_detected', {
                        "alert_id": new_alert.id, 
                        "type": "화재", 
                        "address": self.cctv_name,
                        "lat": float(self.lat), 
                        "lng": float(self.lng),
                        "video_origin": "realtime_its",
                        "is_simulation": False,
                        "image_url": f"/static/captures/{filename}"
                    })
                print(f"🔥 [화재 알람 완료] {self.cctv_name} - ID:{new_alert.id}")

        except Exception as e:
            self.db.session.rollback()
            print(f"❌ 화재 비동기 저장 에러: {e}")

    def run(self):
        """[Main Thread] 데모 코드의 실시간 성능 최적화가 적용된 분석 루프"""
        print(f"🔥 [{self.cctv_name}] 실시간 최적화 분석 시작")

        while self.is_running and self.cap.isOpened():
            # 1. [핵심] 누적된 버퍼 비우기 (실시간 지연 방지)
            # 낡은 프레임은 버리고 무조건 최신 프레임으로 점프합니다.
            for _ in range(3): 
                self.cap.grab()

            success, frame = self.cap.read()
            if not success:
                print(f"📡 [{self.cctv_name}] 연결 재시도...")
                self.cap.open(self.url)
                time.sleep(1)
                continue

            # 2. [최적화] YOLO 추론 설정
            # half=True(FP16 가속), imgsz=320(해상도 축소)로 속도를 극대화합니다.
            # results = self.model.predict(
            #     frame, 
            #     verbose=False, 
            #     conf=0.45, 
            #     imgsz=320, 
            #     # half=True  # GPU 전용 가속 (성능 향상의 핵심)
            # )
            results = self.model.predict(
                frame,
                conf=0.45,
                imgsz=320,
                verbose=False
            )
            
            fire_found_this_frame = False
            
            # 3. [시각화] 결과 처리
            if len(results[0].boxes) > 0:
                fire_found_this_frame = True
                for box in results[0].boxes:
                    b = box.xyxy[0].cpu().numpy()
                    # 빨간색 박스 그리기
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                    cv2.putText(frame, "FIRE", (int(b[0]), int(b[1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 4. 트리거 로직 (DB/파일 저장용 큐 전송)
            if fire_found_this_frame and not self.is_alerting:
                self.alert_queue.put((frame.copy(), datetime.now()))
                self.is_alerting = True
            elif not fire_found_this_frame:
                self.is_alerting = False

            # 5. 최신 프레임 업데이트 (웹 스트리밍용)
            with self.frame_lock:
                self.latest_frame = frame
        
            
            # 6. 미세 대기 (CPU 과부하 방지)
            time.sleep(0.08)

    def stop(self):
        """자원 해제 및 스레드 종료"""
        super().stop()  # 부모의 is_running = False 호출
        if self.cap.isOpened():
            self.cap.release()