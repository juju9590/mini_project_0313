# =============================================================================
# fire_detector.py — 화재 탐지 추론 전용 모듈
# =============================================================================
# 파일명: fire_detector.py
# 저장 위치: N:\개인\이수빈\3.13_Mini_Project\scripts\
# 목적: 웹 서버/프론트팀이 화재 탐지 모델을 실시간으로 호출할 수 있는 모듈
# 사용법: from fire_detector import FireDetector
# =============================================================================

import numpy as np                                  # 배열 연산 (프레임 검증용)
import torch                                        # PyTorch (GPU 확인용)
from ultralytics import YOLO                        # YOLOv8 모델 로드 및 추론
from pathlib import Path                            # 파일 경로 처리


class FireDetector:
    """
    화재 탐지 추론 클래스

    - 모델 로드 (초기화 시 1회)
    - 프레임 입력 → 클래스별 Threshold 후처리 → 연속 프레임 필터 → 알람 판정
    - detect(frame) 호출만으로 알람 여부 + 탐지 결과 반환

    사용 예시:
        detector = FireDetector("path/to/best.pt")
        result = detector.detect(frame)
        if result["alarm"]:
            send_alert()
    """

    # ── 기본 설정값 (클래스 상수) ────────────────────────────────────
    DEFAULT_FIRE_THRESHOLD = 0.10                   # fire 후처리 기준 (미탐 방지)
    DEFAULT_SMOKE_THRESHOLD = 0.25                  # smoke 후처리 기준 (오탐 감소)
    DEFAULT_CONF_THRESHOLD = 0.10                   # YOLO 추론 최소 conf
    DEFAULT_CONSECUTIVE_FRAMES = 10                 # 연속 프레임 필터 기준
    DEFAULT_IMGSZ = 640                             # YOLO 입력 이미지 크기

    def __init__(
        self,
        model_path,                                 # 모델 가중치 파일 경로 (필수)
        fire_threshold=None,                        # fire 후처리 기준 (None이면 기본값)
        smoke_threshold=None,                       # smoke 후처리 기준 (None이면 기본값)
        conf_threshold=None,                        # YOLO 추론 최소 conf (None이면 기본값)
        consecutive_frames=None,                    # 연속 프레임 기준 (None이면 기본값)
        imgsz=None,                                 # 입력 이미지 크기 (None이면 기본값)
        device=None,                                # "cuda" / "cpu" / None(자동 선택)
    ):
        """
        FireDetector 초기화 — 모델 로드 + 설정값 적용

        Parameters:
            model_path (str or Path): YOLOv8 모델 가중치 경로 (.pt 파일)
            fire_threshold (float): fire 클래스 후처리 기준 (기본 0.10)
            smoke_threshold (float): smoke 클래스 후처리 기준 (기본 0.25)
            conf_threshold (float): YOLO 추론 최소 confidence (기본 0.10)
            consecutive_frames (int): 연속 프레임 필터 기준 (기본 10)
            imgsz (int): YOLO 입력 이미지 크기 (기본 640)
            device (str): 추론 장치 — "cuda" / "cpu" / None(자동)
        """

        # ── 1. 모델 경로 검증 ────────────────────────────────────────
        self._model_path = Path(model_path)         # Path 객체로 변환
        if not self._model_path.exists():           # 파일 존재 확인
            raise FileNotFoundError(                # 없으면 에러 발생
                f"모델 파일을 찾을 수 없습니다: {self._model_path}"
            )

        # ── 2. 장치 설정 (GPU 자동 감지) ─────────────────────────────
        if device is not None:                      # 직접 지정했으면
            self._device = device                   # 그대로 사용
        elif torch.cuda.is_available():             # GPU 사용 가능하면
            self._device = 0                        # 첫 번째 GPU (정수 0)
        else:                                       # GPU 없으면
            self._device = "cpu"                    # CPU 사용

        # ── 3. Threshold 설정 ────────────────────────────────────────
        self._fire_threshold = fire_threshold if fire_threshold is not None \
            else self.DEFAULT_FIRE_THRESHOLD        # fire 기준값
        self._smoke_threshold = smoke_threshold if smoke_threshold is not None \
            else self.DEFAULT_SMOKE_THRESHOLD       # smoke 기준값
        self._conf_threshold = conf_threshold if conf_threshold is not None \
            else self.DEFAULT_CONF_THRESHOLD        # YOLO 추론 기준값
        self._consecutive_frames = consecutive_frames if consecutive_frames is not None \
            else self.DEFAULT_CONSECUTIVE_FRAMES    # 연속 프레임 기준
        self._imgsz = imgsz if imgsz is not None \
            else self.DEFAULT_IMGSZ                 # 입력 이미지 크기

        # ── 4. 모델 로드 ─────────────────────────────────────────────
        try:
            self._model = YOLO(str(self._model_path))  # YOLO 모델 로드
        except Exception as e:                      # 로드 실패 시
            raise RuntimeError(                     # 런타임 에러 발생
                f"모델 로드 실패: {e}"
            )

        # ── 5. 클래스 매핑 생성 ──────────────────────────────────────
        self._class_names = self._model.names       # {0: "fire", 1: "smoke"} 딕셔너리
        self._class_thresholds = {}                 # 클래스별 Threshold 매핑
        for idx, name in self._class_names.items(): # 모든 클래스 순회
            if name == "fire":                      # fire이면
                self._class_thresholds[idx] = self._fire_threshold
            elif name == "smoke":                   # smoke이면
                self._class_thresholds[idx] = self._smoke_threshold
            else:                                   # 기타 클래스
                self._class_thresholds[idx] = self._conf_threshold

        # ── 6. 연속 프레임 필터 상태 초기화 ──────────────────────────
        self._consecutive_count = 0                 # 현재 연속 탐지 프레임 수
        self._alarm_active = False                  # 알람 활성 상태

        # ── 7. 초기화 완료 로그 ──────────────────────────────────────
        print(f"🔥 FireDetector 초기화 완료")       # 완료 메시지
        print(f"   모델: {self._model_path.name}")  # 모델 파일명
        print(f"   장치: {self._device}")            # 장치
        print(f"   클래스: {self._class_names}")     # 클래스 목록
        print(f"   fire threshold: {self._fire_threshold}")    # fire 기준
        print(f"   smoke threshold: {self._smoke_threshold}")  # smoke 기준
        print(f"   연속 프레임: {self._consecutive_frames}")    # 연속 기준
        print(f"   imgsz: {self._imgsz}")            # 입력 크기

    def detect(self, frame):
        """
        프레임 1장을 입력받아 화재 탐지 결과를 반환

        Parameters:
            frame (numpy.ndarray): OpenCV BGR 프레임 (H, W, 3)

        Returns:
            dict: {
                "alarm": bool,          # 알람 발동 여부
                "detections": list,     # 탐지 목록 [{class, confidence, bbox}]
                "consecutive_count": int # 현재 연속 탐지 프레임 수
            }
        """

        # ── 1. 입력 검증 ─────────────────────────────────────────────
        if frame is None:                           # 프레임이 None이면
            return self._make_result(               # 빈 결과 반환
                alarm=False, detections=[], count=self._consecutive_count
            )

        if not isinstance(frame, np.ndarray):       # numpy 배열이 아니면
            raise TypeError(                        # 타입 에러 발생
                f"frame은 numpy.ndarray여야 합니다. 받은 타입: {type(frame)}"
            )

        if frame.ndim != 3 or frame.shape[2] != 3: # 3채널이 아니면
            raise ValueError(                       # 값 에러 발생
                f"frame은 (H, W, 3) 형태여야 합니다. 받은 shape: {frame.shape}"
            )

        # ── 2. YOLO 추론 (conf 낮게 → 일단 다 받기) ─────────────────
        results = self._model.predict(              # 모델 예측
            frame,                                  # 입력 프레임
            conf=self._conf_threshold,              # 낮은 conf로 다 받기
            imgsz=self._imgsz,                      # 입력 크기
            device=self._device,                    # GPU or CPU
            verbose=False                           # 로그 끄기
        )

        # ── 3. 클래스별 Threshold 후처리 ─────────────────────────────
        boxes = results[0].boxes                    # 바운딩박스 객체
        detections = []                             # 후처리 통과 탐지 리스트

        for box in boxes:                           # 각 bbox 순회
            cls_id = int(box.cls[0])                # 클래스 번호
            conf = float(box.conf[0])               # 신뢰도
            cls_name = self._class_names.get(       # 클래스 이름
                cls_id, f"unknown_{cls_id}"
            )
            threshold = self._class_thresholds.get( # 해당 클래스 threshold
                cls_id, self._conf_threshold
            )

            if conf >= threshold:                   # threshold 이상이면 유지
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()  # 좌표 추출
                detections.append({                 # 탐지 결과 추가
                    "class": cls_name,              # 클래스명
                    "confidence": round(conf, 4),   # 신뢰도 (소수점 4자리)
                    "bbox": [                       # 바운딩박스 좌표 (정수)
                        int(x1), int(y1),           # 좌상단 (x1, y1)
                        int(x2), int(y2)            # 우하단 (x2, y2)
                    ]
                })

        # ── 4. 연속 프레임 필터 ───────────────────────────────────────
        detected = len(detections) > 0              # 후처리 후 탐지 여부

        if detected:                                # 탐지됐으면
            self._consecutive_count += 1            # 연속 카운터 증가
        else:                                       # 탐지 안 됐으면
            self._consecutive_count = 0             # 카운터 리셋
            self._alarm_active = False              # 알람 비활성화

        # ── 5. 알람 판정 ──────────────────────────────────────────────
        alarm = False                               # 알람 기본값 = False
        if self._consecutive_count >= self._consecutive_frames:  # 기준 도달
            if not self._alarm_active:              # 새 알람이면
                self._alarm_active = True           # 알람 활성화
            alarm = True                            # 알람 발동

        # ── 6. 결과 반환 ──────────────────────────────────────────────
        return self._make_result(                   # 결과 딕셔너리 생성
            alarm=alarm,                            # 알람 여부
            detections=detections,                  # 탐지 목록
            count=self._consecutive_count           # 연속 카운터
        )

    def reset(self):
        """
        연속 프레임 카운터 초기화
        — 새 영상 시작 시 또는 수동 리셋 시 호출
        """
        self._consecutive_count = 0                 # 카운터 리셋
        self._alarm_active = False                  # 알람 비활성화

    # ── 읽기 전용 속성 (프로퍼티) ─────────────────────────────────────

    @property
    def consecutive_count(self):
        """현재 연속 탐지 프레임 수 (읽기 전용)"""
        return self._consecutive_count              # 현재 카운터 반환

    @property
    def is_alarm_active(self):
        """알람 활성 상태 (읽기 전용)"""
        return self._alarm_active                   # 알람 상태 반환

    @property
    def class_names(self):
        """모델 클래스 목록 (읽기 전용)"""
        return self._class_names                    # 클래스 딕셔너리 반환

    @property
    def config(self):
        """현재 설정값 딕셔너리 (읽기 전용)"""
        return {                                    # 설정값 반환
            "model_path": str(self._model_path),    # 모델 경로
            "device": str(self._device),            # 장치
            "fire_threshold": self._fire_threshold, # fire 기준
            "smoke_threshold": self._smoke_threshold,  # smoke 기준
            "conf_threshold": self._conf_threshold, # YOLO 추론 기준
            "consecutive_frames": self._consecutive_frames,  # 연속 기준
            "imgsz": self._imgsz,                   # 입력 크기
            "class_names": self._class_names         # 클래스 목록
        }

    # ── 내부 헬퍼 메서드 ──────────────────────────────────────────────

    @staticmethod
    def _make_result(alarm, detections, count):
        """결과 딕셔너리 생성 헬퍼"""
        return {                                    # 표준 결과 포맷
            "alarm": alarm,                         # 알람 여부 (bool)
            "detections": detections,               # 탐지 목록 (list)
            "consecutive_count": count              # 연속 카운터 (int)
        }


# =============================================================================
# 사용 예시 — 시스템팀이 아래처럼 호출하면 됩니다
# =============================================================================
if __name__ == "__main__":

    import cv2                                      # OpenCV (영상 읽기)

    # ── 1. 모델 경로 설정 ─────────────────────────────────────────
    MODEL_PATH = r"N:\개인\이수빈\3.13_Mini_Project\results\yolov8n_tuned\weights\best.pt"

    # ── 2. FireDetector 초기화 (모델 로드 — 1회만) ────────────────
    detector = FireDetector(MODEL_PATH)             # 기본 설정으로 초기화
    # 또는 설정값을 직접 지정:
    # detector = FireDetector(
    #     MODEL_PATH,
    #     fire_threshold=0.10,                      # fire 기준
    #     smoke_threshold=0.25,                     # smoke 기준
    #     consecutive_frames=10,                    # 연속 프레임 기준
    #     device="cuda"                             # GPU 강제 지정
    # )

    # ── 3. 영상 열기 ──────────────────────────────────────────────
    VIDEO_PATH = r"N:\개인\이수빈\3.13_Mini_Project\DATASET\테스트영상\2.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)              # 영상 열기

    if not cap.isOpened():                          # 열기 실패
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
    else:
        frame_count = 0                             # 프레임 카운터
        alarm_count = 0                             # 알람 카운터

        while True:                                 # 프레임 루프
            ret, frame = cap.read()                 # 프레임 읽기
            if not ret:                             # 영상 끝
                break                               # 루프 종료

            # ── 4. detect() 호출 — 핵심 1줄 ──────────────────────
            result = detector.detect(frame)         # 탐지 실행

            # ── 5. 알람 처리 ──────────────────────────────────────
            if result["alarm"]:                     # 알람 발동 시
                alarm_count += 1                    # 알람 카운터 증가
                det_summary = ", ".join(             # 탐지 요약 문자열
                    [f'{d["class"]}({d["confidence"]:.2f})' for d in result["detections"]]
                )
                print(f"🚨 알람 #{alarm_count} | 프레임 {frame_count} | "
                      f"연속 {result['consecutive_count']} | {det_summary}")

                # ★ 여기에 실제 알람 로직 추가:
                # send_alert()                      # 알림 전송
                # save_snapshot(frame)               # 스냅샷 저장
                # log_to_db(result)                  # DB 기록

            frame_count += 1                        # 프레임 카운터 증가

            # 진행률 (1000프레임마다)
            if frame_count % 1000 == 0:             # 1000프레임마다
                print(f"   [{frame_count:,}프레임] "
                      f"연속: {result['consecutive_count']} | 알람: {alarm_count}회")

        cap.release()                               # 영상 닫기

        # ── 6. 결과 출력 ──────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"📊 테스트 완료")
        print(f"   총 프레임: {frame_count:,}")
        print(f"   총 알람: {alarm_count}회")
        print(f"   설정: {detector.config}")
