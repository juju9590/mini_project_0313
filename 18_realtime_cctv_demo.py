# =============================================================================
# 18_realtime_cctv_demo.py
# 실행 방법: 명령프롬프트에서 python 18_realtime_cctv_demo.py
# 종료: 'q' 키
# =============================================================================


# 실행하면 이렇게 진행

# 모델 자동 로드
# 서울 지역 CCTV 목록 조회 → 번호 목록 출력
# "CCTV 번호를 입력하세요" → 번호 타이핑
# 연결 테스트
# 실시간 탐지 창이 뜸 → q키로 종료

# =============================================================================

# 수정할 곳 (파일 상단)
# 항목                        변수                             설명
# API 키                   API_KEY본인                      ITS 인증키
# 녹화 여부                  RECORD                         True/False
# 실행 시간                MAX_SECONDS                   120=2분, None=무한
# 지역 변경        MIN_X, MAX_X, MIN_Y, MAX_Y             위도/경도 조정

# =============================================================================

# YOLO 모델
from ultralytics import YOLO

# 영상 처리
import cv2

# HTTP 요청 (API 호출)
import requests

# 경로 처리
from pathlib import Path

# 숫자 계산
import numpy as np

# 시간 측정
import time
from datetime import datetime

# JSON 처리
import json

# =============================================================================
# ★★★ 설정 — 여기만 수정하세요 ★★★
# =============================================================================

# ITS 국가교통정보센터 API 키
API_KEY = "2040bbf03af04cf7b83d1841b06ef78e"

# 프로젝트 경로
PROJECT_ROOT = Path(r'N:\개인\이수빈\3.13_Mini_Project')

# 모델 경로
MODEL_PATH = PROJECT_ROOT / 'results' / 'yolov8n_tuned' / 'weights' / 'best.pt'

# 녹화 저장 폴더
DEMO_DIR = PROJECT_ROOT / 'evaluation' / 'realtime_demo'
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Threshold
CONF_THRESHOLD = 0.10

# 녹화 여부 (True면 mp4로 저장)
RECORD = True

# 최대 실행 시간 (초, None이면 q키로만 종료)
MAX_SECONDS = 120

# 지역 좌표 (서울 기본값 — 다른 지역은 좌표 변경)
# MIN_X = '126.8'
# MAX_X = '127.2'
# MIN_Y = '37.4'
# MAX_Y = '37.7'

# 서해안고속도로 (서쪽 석양 방향)
MIN_X = '126.5'
MAX_X = '127.0'
MIN_Y = '37.2'
MAX_Y = '37.6'

# ITS API URL
ITS_API_URL = "https://openapi.its.go.kr:9443/cctvInfo"

# =============================================================================
# CCTV 목록 조회
# =============================================================================

def get_cctv_list(api_key, min_x, max_x, min_y, max_y, road_type='its'):
    """ITS API로 CCTV 목록 조회 (its=국도, ex=고속도로)"""
    params = {
        'apiKey': api_key,
        'type': road_type,
        'cctvType': 1,       # 1=실시간 스트리밍
        'minX': min_x,
        'maxX': max_x,
        'minY': min_y,
        'maxY': max_y,
        'getType': 'json'
    }
    try:
        response = requests.get(ITS_API_URL, params=params, timeout=10)
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            return data['response']['data']
        return []
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return []

# =============================================================================
# 실시간 탐지 함수
# =============================================================================

def run_realtime_detection(model, stream_url, cctv_name,
                           conf_threshold=0.10,
                           record=False,
                           record_path=None,
                           max_seconds=None):
    """실시간 CCTV 화재 탐지"""

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ 스트림 연결 실패")
        return None

    # 영상 정보
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print(f"🔴 실시간 화재 탐지 시작")
    print(f"   CCTV: {cctv_name}")
    print(f"   해상도: {w}x{h}")
    print(f"   Threshold: {conf_threshold}")
    if record:
        print(f"   녹화: {record_path}")
    print(f"   종료: 'q' 키")
    print(f"{'='*60}")

    # 창 생성
    cv2.namedWindow('Fire Detection - CCTV', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fire Detection - CCTV', min(w, 1280), min(h, 720))

    # 녹화 설정
    writer = None
    if record and record_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(record_path), fourcc, 15, (w, h))

    # 통계
    total_frames = 0
    detected_frames = 0
    fps_list = []
    start_time_total = time.time()
    fire_detected_ever = False

    while True:
        # 시간 제한
        if max_seconds and (time.time() - start_time_total) > max_seconds:
            print(f"\n⏱️ {max_seconds}초 경과 — 자동 종료")
            break

        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임 읽기 실패 — 재연결 시도...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(stream_url)
            continue

        # 추론
        t_start = time.time()
        results = model.predict(
            frame,
            conf=conf_threshold,
            imgsz=640,
            save=False,
            verbose=False
        )
        t_end = time.time()

        infer_ms = (t_end - t_start) * 1000
        current_fps = 1000 / infer_ms if infer_ms > 0 else 0
        fps_list.append(current_fps)

        # 탐지 결과
        num_boxes = len(results[0].boxes)
        detected = num_boxes > 0

        if detected:
            detected_frames += 1
            fire_detected_ever = True
            # bbox 그리기 (YOLO 내장)
            display_frame = results[0].plot()
        else:
            display_frame = frame.copy()

        # 상단 정보 바
        bar_color = (0, 0, 200) if detected else (50, 50, 50)
        cv2.rectangle(display_frame, (0, 0), (w, 45), bar_color, -1)

        # CCTV 이름
        cv2.putText(display_frame, f"CCTV: {cctv_name[:30]}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (w - 120, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 탐지 상태
        status = "FIRE DETECTED!" if detected else "Monitoring..."
        status_color = (0, 0, 255) if detected else (200, 200, 200)
        cv2.putText(display_frame, status, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Threshold
        cv2.putText(display_frame, f"Threshold: {conf_threshold}", (w - 180, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 녹화 표시
        if record:
            cv2.circle(display_frame, (w - 20, 12), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (w - 55, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 화면 출력
        cv2.imshow('Fire Detection - CCTV', display_frame)

        # 녹화
        if writer:
            writer.write(display_frame)

        total_frames += 1

        # 'q' 키 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 'q' 키로 종료")
            break

    # 정리
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # 통계
    elapsed = time.time() - start_time_total
    avg_fps = np.mean(fps_list) if fps_list else 0

    stats = {
        'cctv_name': cctv_name,
        'total_frames': total_frames,
        'detected_frames': detected_frames,
        'detection_rate': round(detected_frames / total_frames * 100, 1) if total_frames > 0 else 0,
        'avg_fps': round(avg_fps, 1),
        'elapsed_seconds': round(elapsed, 1),
        'fire_detected': fire_detected_ever
    }

    print(f"\n📊 실행 통계:")
    print(f"   총 프레임: {total_frames}")
    print(f"   탐지 프레임: {detected_frames} ({stats['detection_rate']}%)")
    print(f"   평균 FPS: {avg_fps:.1f}")
    print(f"   실행 시간: {elapsed:.1f}초")
    print(f"   화재 탐지: {'🔥 있음' if fire_detected_ever else '✅ 없음 (정상)'}")

    return stats

# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🔥 공공 CCTV 실시간 화재 탐지 데모")
    print("=" * 60)

    # 1. 모델 로드
    print("\n📦 모델 로드 중...")
    model = YOLO(str(MODEL_PATH))
    print(f"✅ 모델 로드 완료: {MODEL_PATH.name}")

    # 2. CCTV 목록 조회
    print("\n📡 CCTV 목록 조회 중...")
    its_cctvs = get_cctv_list(API_KEY, MIN_X, MAX_X, MIN_Y, MAX_Y, 'its')
    ex_cctvs = get_cctv_list(API_KEY, MIN_X, MAX_X, MIN_Y, MAX_Y, 'ex')
    all_cctvs = its_cctvs + ex_cctvs
    print(f"총 {len(all_cctvs)}개 CCTV 발견")

    if len(all_cctvs) == 0:
        print("❌ CCTV를 찾지 못했습니다. API 키와 좌표를 확인하세요.")
        exit()

    # 3. CCTV 목록 출력
    print(f"\n{'번호':<6} {'CCTV명':<40}")
    print("-" * 50)
    for i, cctv in enumerate(all_cctvs[:30]):
        name = cctv.get('cctvname', '이름없음')[:38]
        print(f"{i:<6} {name}")
    if len(all_cctvs) > 30:
        print(f"... 외 {len(all_cctvs) - 30}개")

    # 4. CCTV 선택
    while True:
        try:
            idx = int(input(f"\n▶ CCTV 번호를 입력하세요 (0~{len(all_cctvs)-1}): "))
            if 0 <= idx < len(all_cctvs):
                break
            print("범위를 벗어났습니다.")
        except ValueError:
            print("숫자를 입력하세요.")

    selected = all_cctvs[idx]
    cctv_name = selected.get('cctvname', '이름없음')
    cctv_url = selected.get('cctvurl', '')

    print(f"\n선택: {cctv_name}")

    # 5. 연결 테스트
    print("🔌 연결 테스트 중...")
    test_cap = cv2.VideoCapture(cctv_url)
    if not test_cap.isOpened():
        print("❌ 연결 실패 — 다른 CCTV를 선택해주세요")
        test_cap.release()
        exit()
    ret, _ = test_cap.read()
    test_cap.release()
    if not ret:
        print("❌ 프레임 읽기 실패 — 다른 CCTV를 선택해주세요")
        exit()
    print("✅ 연결 성공!")

    # 6. 녹화 파일 경로
    record_path = None
    if RECORD:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = cctv_name.replace(' ', '_').replace('/', '_')[:20]
        record_path = DEMO_DIR / f'demo_{timestamp}_{safe_name}.mp4'

    # 7. 실시간 탐지 실행
    stats = run_realtime_detection(
        model=model,
        stream_url=cctv_url,
        cctv_name=cctv_name,
        conf_threshold=CONF_THRESHOLD,
        record=RECORD,
        record_path=record_path,
        max_seconds=9000
    )

    # 8. 결과 저장
    if stats:
        result_data = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "YOLOv8n_tuned",
            "threshold": CONF_THRESHOLD,
            "cctv_name": cctv_name,
            "stats": stats
        }
        json_path = DEMO_DIR / 'realtime_demo_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장: {json_path}")
        if RECORD and record_path and record_path.exists():
            size_mb = record_path.stat().st_size / (1024 * 1024)
            print(f"💾 녹화 파일: {record_path} ({size_mb:.1f} MB)")

    print("\n✅ 데모 종료!")


# 31번추천