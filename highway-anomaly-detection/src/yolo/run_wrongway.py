# 스크립트 실행 진입점
# 설정을 구성하고 Detector를 생성/실행

from pathlib import Path
from wrongway import Detector, DetectorConfig
import sys

# 프로젝트 루트 경로 (이 파일 위치 기준)
ROOT = Path(r"N:\개인\박대원\0211~0313_miniproject\highway-anomaly-detection")

# wrongway 패키지가 들어있는 폴더 (src/yolo)
YOLO_DIR = ROOT / "src" / "yolo"

# import 경로 추가 (어디서 실행해도 from wrongway import ... 되게)
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))
    
cfg = DetectorConfig(
    model_path=ROOT / "runs" / "yolo11n_vehicle_v5" / "weights" / "best.pt",
    conf=0.7,                          # 신뢰도 임계값
    grid_size=15,                       # Flow Map 그리드 크기
    target_classes=None,                # None이면 YOLO 모델의 모든 클래스를 사용
    enable_online_flow_update=True,     # 정상 흐름을 학습에 사용
    detect_only=True,                  # 기존 flow_map 로드 후 탐지 전용
    log_dir=ROOT / "logs", # 여기 아래에 CSV 3개 생성됨
    flow_map_path=ROOT / "models" / "flow_map.npy",
    result_dir=ROOT / "results",
    data_dir=ROOT / "data"         
)

if __name__ == "__main__":
    detector = Detector(cfg)
    detector.run("화면전환_테스트.mp4")  # 지정한 비디오 파일로 실행