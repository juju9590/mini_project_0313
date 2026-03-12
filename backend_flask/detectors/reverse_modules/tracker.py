# YOLO + ByteTrack 래퍼
# OpenVINO 가속 모델 객체를 주입받아 track() 결과를 정리된 dict 리스트로 반환

class YoloTracker:
    def __init__(self, model_instance, conf, target_classes=None):
        """
        Args:
            model_instance: 이미 로드된 YOLO(OpenVINO) 모델 객체
            conf (float): 객체 검출 신뢰도 임계값
            target_classes (list): 추적할 클래스 인덱스 리스트 (ex: [2, 3, 5, 7])
        """
        # [수정] 경로가 아닌 로드된 모델 객체를 그대로 저장합니다.
        self.model = model_instance 
        self.conf = conf
        self.target_classes = target_classes

        # 모델에서 클래스 이름 자동 로드 (ex: {0: 'car', 1: 'bus', ...})
        self.class_names = getattr(self.model, "names", {})

    def track(self, frame):
        """
        YOLO 추적 실행 후 결과를 정리된 dict 리스트로 반환
        Returns:
            list of dict: [{id, x1, y1, x2, y2, cx, cy}, ...]
        """
        # YOLO 추적 설정
        track_kwargs = {
            "tracker": "bytetrack.yaml",   # ByteTrack 설정
            "persist": True,               # 트랙 ID 유지
            "verbose": False,              # 로그 최소화
            "conf": self.conf,             # 신뢰도 임계값
            "imgsz": 320,                  # OpenVINO 환경에서 속도 최적화를 위해 320 권장
        }
        
        # 특정 클래스만 필터링할 경우 추가
        if self.target_classes is not None:
            track_kwargs["classes"] = self.target_classes

        # [핵심] OpenVINO 가속이 적용된 모델 객체의 track 메서드 호출
        results = self.model.track(frame, **track_kwargs)
        
        # 감지된 객체나 트랙 ID가 없으면 빈 리스트 반환
        if results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes
        tracks = []
        
        # ID와 좌표 리스트 추출 (tensor 환경 고려하여 cpu().tolist() 사용)
        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.cpu().tolist()

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = coords[i]
            tracks.append({
                "id": tid,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
            })
            
        return tracks