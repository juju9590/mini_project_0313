# 모든 고정 파라미터를 한 곳에서 관리하는 설정 클래스

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectorConfig:
    # ==================== 모델/기본 설정 ====================
    model_path: str | Path = ""       # YOLO 모델 경로
    conf: float = 0.66                # 객체 검출 신뢰도(confidence) 임계값
    target_classes: list | None = None  # 추적할 클래스 인덱스 리스트 (None이면 모든 클래스)

    # ==================== 흐름 그리드(Flow Map) 설정 ====================
    grid_size: int = 15               # 흐름 맵을 나눌 격자 크기 (N x N)

    # ==================== 학습(learning) 관련 설정 ====================
    learning_frames: int = 500        # 초기 학습에 사용할 프레임 수 (Flow Map 학습 전용)
    alpha: float = 0.10               # EMA 학습 속도 (새 데이터 반영 비율 10%)
    min_samples: int = 5              # 셀당 최소 학습 샘플 수 (이하이면 공간 보정에 사용)
    enable_online_flow_update: bool = False # 정상 흐름 학습에 사용(True)

    # ==================== 역주행 탐지 관련 설정 ====================
    velocity_window: int = 15         # 속도/방향 계산 시 사용하는 프레임 간격 (이전 위치~현재 위치 거리)
    base_speed_threshold: float = 7.0 # 기본 속도 임계값 (원근에 따라 가중을 곱해 사용)
    cos_threshold: float = -0.5       # 코사인 유사도 임계값 (이하이면 흐름과 반대 방향으로 간주)
    wrong_count_threshold: int = 4    # 역주행 확정까지 필요한 연속 의심 횟수
    vote_threshold: float = 0.6       # 투표 시 역방향 비율 임계값 (60% 이상이면 역주행 의심)
    min_move_distance: float = 20.0   # 최소 누적 이동 거리 (이하면 정지로 판단)
    min_move_per_frame: float = 1.5   # 프레임당 평균 이동거리 (이하면 정지)

    # ==================== ID 매핑 관련 ====================
    id_match_distance: int = 120      # ID 재매칭 허용 거리 (픽셀 단위, 이전 ID와 새 ID 위치 비교)
    trail_length: int = 30            # 차량 궤적 최대 길이 (리스트에 최근 몇 점까지 보관할지)
    stale_threshold: int = 90         # 안 보이는 ID를 삭제하기까지의 프레임 수
    reappear_frame_limit: int = 45    # ID 재매칭 시 사라진 지 최대 몇 프레임까지 허용
    last_pos_expire: int = 60         # 역주행 마지막 위치 기록 만료 프레임

    # ==================== 카메라 전환 감지 관련 ====================
    relearn_frames: int = 300         # 재학습에 사용할 프레임 수
    cooldown_frames: int = 150        # 재학습 후 전환 감지 비활성 프레임 수
    switch_confirm_needed: int = 4    # 전환으로 확정하기 위해 필요한 연속 감지 횟수

    # ==================== 경로 관련 ====================
    flow_map_path: Path = None        # flow_map 저장/로드 파일 경로
    result_dir: Path = None           # 결과 영상 저장 폴더 경로
    data_dir: Path = None             # 입력 데이터(영상) 폴더 경로
    
    # ==================== 로깅/실행 모드 ====================
    detect_only: bool = True            # True면 flow_map 필수(학습 안 함)
    log_dir: Path | None = None         # 로그 저장 폴더
    log_interval_frames: int = 5        # 트랙/프레임 로그를 N프레임마다 기록