import cv2
import numpy as np
import os
import time
import threading
from queue import Queue
from datetime import datetime
from collections import defaultdict
from .base_detector import BaseDetector  # BaseDetector가 같은 폴더에 있다고 가정
from yolo_models.yolo_model import yolo_reverse

from .reverse_modules.config import DetectorConfig
from .reverse_modules.tracker import YoloTracker
from .reverse_modules.flow_map import FlowMap
from .reverse_modules.judge import WrongWayJudge
from .reverse_modules.bbox_stabilizer import BBoxStabilizer
from .reverse_modules.id_manager import IDManager
from .reverse_modules.camera_switch import CameraSwitchDetector

# 모듈들이 공통으로 상태를 공유할 객체
class State:
    def __init__(self, cfg=None):
        self.frame_num = 0
        self.video_fps = 30
        self.frame_w = 640
        self.frame_h = 360
        self.trajectories = defaultdict(list)
        self.wrong_way_ids = set()
        self.wrong_way_count = defaultdict(int)
        
        self.first_seen_frame = {}
        self.first_suspect_frame = {}
        self.display_id_map = {}
        self.next_wrong_way_label = 1
        self.detection_stats = {}
        self.wrong_way_last_pos = {}
        self._stale_counter = defaultdict(int)

class ReverseDetector(BaseDetector):
    def __init__(self, cctv_name, url, lat=37.5, lng=127.0, socketio=None, db=None, ResultModel=None, ReverseModel=None, conf=None, app=None):
        # 부모 클래스(BaseDetector) 초기화
        super().__init__(cctv_name, url, app=app, socketio=socketio, db=db, ResultModel=ResultModel)
        
        self.lat = lat  
        self.lng = lng 
        self.ReverseModel = ReverseModel
        
        # 환경 변수 및 설정
        conf_val = float(os.getenv('CONFIDENCE_THRESHOLD') or conf or 0.66)
        self.cfg = DetectorConfig(conf=conf_val)
        self.st = State(self.cfg)
        
        # 모델 및 캡처 설정
        # self.model = YOLO("best_DW.pt")
        self.model = yolo_reverse
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 버퍼 최소화
        
        # 1. 트래커 래퍼 (기존 model 로드 대체)
        # target_classes=[2, 3, 5, 7] 은 각각 car, motorcycle, bus, truck (COCO 기준 예시)
        self.tracker = YoloTracker(self.model, conf=self.cfg.conf)
        
        # 2. 흐름장 (Grid 15x15)
        self.flow_map = FlowMap(grid_size=self.cfg.grid_size, alpha=self.cfg.alpha, min_samples=self.cfg.min_samples)
        
        # 3. 판독기
        self.judge = WrongWayJudge(self.cfg, self.flow_map, self.st)
        
        # 4. BBox 흔들림 보정
        self.stabilizer = BBoxStabilizer(alpha=0.5)
        
        # 5. ID 및 라벨 관리
        self.id_manager = IDManager(self.cfg, self.flow_map, self.st)

        self.camera_detector = CameraSwitchDetector(self.cfg)
        
        # 상태 제어 변수
        self.prev_frame_gray = None
        self.SCENE_THRESHOLD = 30.0
        self.VELOCITY_WINDOW = 5
        self.LEARNING_FRAMES = 150
        self.learning_done = False
        self.alerted_ids = set()

        # 모델 세이브/로드 설정
        safe_name = cctv_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        self.save_dir = "learned_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_file = os.path.join(self.save_dir, f"flow_{safe_name}.npy")
        
        # 학습된 데이터 로드
        self.load_flow_map()

    def load_flow_map(self):
        """저장된 이동 경로 맵 로드 (FlowMap 클래스의 load 메서드 활용)"""
        # self.model_file은 이미 Path 객체이거나 문자열일 것이므로 바로 전달
        from pathlib import Path
        model_path = Path(self.model_file)

        # 1. FlowMap 내부의 load 메서드를 사용하여 데이터 로드
        # 이 방식이 FlowMap 객체의 메서드(learn_step 등)를 보존하는 가장 안전한 방법입니다.
        success = self.flow_map.load(model_path)
        
        if success:
            self.learning_done = True
            # 로드된 데이터 기반으로 상태 값 동기화
            self.grid_h, self.grid_w = self.flow_map.flow.shape[:2]
            print(f"✅ [{self.cctv_name}] 학습 모델 로드 완료 및 적용")
        else:
            self.learning_done = False
            print(f"⚠️ [{self.cctv_name}] 로드할 모델이 없거나 실패하여 신규 학습 모드로 시작합니다.")

    def save_flow_map(self):
        """학습된 이동 경로 맵 저장"""
        if self.flow_map is not None:
            np.save(self.model_file, {"flow": self.flow_map, "count": self.flow_count})
            print(f"💾 [{self.cctv_name}] 학습 완료 및 모델 저장됨")

    def process_alert(self, data):
        """[Worker Thread] 비동기 DB 저장 및 알림 처리"""
        frame, alert_time, track_id = data
        try:
            with self.app.app_context():
                # 1. 공통 결과 테이블 저장 (DetectionResult)
                new_alert = self.ResultModel(
                    event_type="reverse", address=self.cctv_name,
                    latitude=self.lat, longitude=self.lng,
                    detected_at=alert_time, is_simulation=False, is_resolved=False
                )
                self.db.session.add(new_alert)
                self.db.session.flush()

                # 2. 이미지 파일 저장
                ts = alert_time.strftime("%Y%m%d_%H%M%S")
                filename = f"reverse_real_{new_alert.id}_{ts}.jpg"
                save_path = os.path.join(self.app.root_path, "static", "captures")
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, filename)
                cv2.imwrite(filepath, frame)

                # 3. 역주행 상세 테이블 저장 (ReverseResult)
                from models import ReverseResult
                reverse_detail = ReverseResult(
                    result_id=new_alert.id,
                    image_path=f"/static/captures/{filename}",
                    vehicle_info=f"ID:{track_id} 실시간 탐지"
                )
                self.db.session.add(reverse_detail)
                self.db.session.commit()

                # 4. 소켓 알림 전송
                if self.socketio:
                    self.socketio.emit('anomaly_detected', {
                        "alert_id": new_alert.id, "type": "역주행", 
                        "address": self.cctv_name, "lat": float(self.lat), "lng": float(self.lng),
                        "video_origin": "realtime_its", "is_simulation": False,
                        "image_url": f"/static/captures/{filename}"
                    })
                print(f"🚨 [역주행 알람 완료] {self.cctv_name} - ID:{track_id}")
        except Exception as e:
            self.db.session.rollback()
            print(f"❌ 역주행 비동기 저장 에러: {e}")

    def run(self):
        """[Main Thread] 백그라운드 분석 루프 - 시각화 후 캡처 로직 적용"""
        print(f"🚗 [{self.cctv_name}] 분석 시작 (박스 포함 캡처 적용)")
        
        while self.is_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (640, 360))
            h, w = frame.shape[:2]
            
            if self.flow_map.frame_w == 0:
                self.st.frame_w, self.st.frame_h = w, h
                self.flow_map.init_grid(w, h)
            self.st.frame_num += 1

            # 1. 카메라 전환 감지
            if self.camera_detector.check(frame, self.st.frame_num, self.cfg.cooldown_frames):
                self.st.frame_num = 0
                self.learning_done = False
                self.flow_map.reset()
                self.st.trajectories.clear()
                continue

            # 2. 객체 추적
            tracks = self.tracker.track(frame)
            active_ids = set()
            pending_alert_id = None  # 이번 프레임에서 알람을 쏠 ID 저장용

            for t in tracks:
                raw_id = t["id"]
                active_ids.add(raw_id)
                
                # 좌표 보정 및 궤적 업데이트
                sx1, sy1, sx2, sy2, cx, cy = self.stabilizer.stabilize(
                    raw_id, (t["x1"], t["y1"], t["x2"], t["y2"]), self.st.frame_num
                )
                self.st.trajectories[raw_id].append((cx, cy))
                if len(self.st.trajectories[raw_id]) > self.cfg.trail_length:
                    self.st.trajectories[raw_id].pop(0)

                # 3. 역주행 판독
                self.id_manager.check_reappear(raw_id, cx, cy)
                traj = self.st.trajectories[raw_id]
                
                if len(traj) >= self.cfg.velocity_window:
                    px, py = traj[-self.cfg.velocity_window]
                    vdx, vdy = cx - px, cy - py
                    speed = np.sqrt(vdx**2 + vdy**2)
                    
                    if speed > self.cfg.min_move_per_frame:
                        ndx, ndy = vdx / (speed + 1e-6), vdy / (speed + 1e-6)
                        
                        if not self.learning_done:
                            self.flow_map.learn_step(px, py, cx, cy, min_move=5)
                        else:
                            # 판독 결과 확인
                            is_wrong, _, _ = self.judge.check(raw_id, traj, ndx, ndy, speed, cy)
                            if is_wrong and raw_id not in self.st.alerted_ids:
                                self.st.alerted_ids.add(raw_id)
                                self.id_manager.assign_label(raw_id)
                                # 🚨 여기서 바로 큐에 넣지 않고 ID만 킵해둡니다.
                                pending_alert_id = raw_id

                # --- 4. 시각화 (이미지에 박스를 먼저 다 그립니다) ---
                # 방금 판독된 ID거나 이미 역주행 확정된 ID면 빨간색
                is_confirmed = (raw_id in self.st.wrong_way_ids) or (raw_id == pending_alert_id)
                color = (0, 0, 255) if is_confirmed else (0, 255, 0)
                label = self.id_manager.get_display_label(raw_id) or f"ID:{raw_id}"
                
                cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color, 2)
                cv2.putText(frame, label, (int(sx1), int(sy1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- 5. 최종 캡처 및 알람 전송 ---
            # 박스가 모두 그려진 frame을 복사하여 큐에 넣습니다.
            if pending_alert_id is not None:
                print(f"🚨 [캡처] 역주행 ID:{pending_alert_id} 시각화 포함 저장 시도")
                self.alert_queue.put((frame.copy(), datetime.now(), pending_alert_id))

            # 6. 정리
            self.stabilizer.cleanup(active_ids)
            self.id_manager.cleanup(active_ids)

            if not self.learning_done and self.st.frame_num >= self.cfg.learning_frames:
                self.flow_map.apply_spatial_smoothing()
                self.learning_done = True
                print(f"✅ [{self.cctv_name}] 학습 완료")

            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(0.01)

    def stop(self):
        super().stop()
        if self.cap.isOpened():
            self.cap.release()