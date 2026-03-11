# 실시간 cctv 데모 + 연속 프레임 후처리


# =============================================================================
# 18_realtime_cctv_demo.py  (v2 — 후처리 로직 + 이미지 자동 저장)
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
# API 키                   API_KEY                        본인 ITS 인증키
# 녹화 여부                  RECORD                         True/False
# 실행 시간                MAX_SECONDS                   120=2분, None=무한
# 지역 변경        MIN_X, MAX_X, MIN_Y, MAX_Y             위도/경도 조정
# 연속 프레임 수       CONSECUTIVE_FRAMES               알람 발생 기준 프레임 수
# 저장 간격           SAVE_INTERVAL_SEC                이미지 저장 간격(초)
# =============================================================================

# YOLO 모델                                              # ultralytics 라이브러리에서 YOLO 클래스 가져오기
from ultralytics import YOLO

# 영상 처리                                              # OpenCV 라이브러리 가져오기
import cv2

# HTTP 요청 (API 호출)                                    # requests 라이브러리 가져오기
import requests

# 경로 처리                                              # pathlib에서 Path 클래스 가져오기
from pathlib import Path

# 숫자 계산                                              # numpy 라이브러리 가져오기
import numpy as np

# 시간 측정                                              # time 모듈 가져오기
import time                                              # 시간 측정용
from datetime import datetime                            # 날짜/시간 포맷용

# JSON 처리                                              # json 모듈 가져오기
import json

# =============================================================================
# ★★★ 설정 — 여기만 수정하세요 ★★★
# =============================================================================

# ITS 국가교통정보센터 API 키                              # 본인 API 키로 교체
API_KEY = "2040bbf03af04cf7b83d1841b06ef78e"

# 프로젝트 경로                                           # 프로젝트 루트 디렉토리 경로
PROJECT_ROOT = Path(r'N:\\개인\\이수빈\\3.13_Mini_Project')

# 모델 경로                                              # YOLOv8n 튜닝 모델 경로
MODEL_PATH = PROJECT_ROOT / 'results' / 'yolov8n_tuned' / 'weights' / 'best.pt'

# 녹화 저장 폴더                                          # 녹화 파일 저장 디렉토리
DEMO_DIR = PROJECT_ROOT / 'evaluation' / 'realtime_demo'
DEMO_DIR.mkdir(parents=True, exist_ok=True)              # 폴더 없으면 자동 생성

# ★ 이미지 저장 폴더 (탐지 시 원본/bbox 이미지 저장)         # 이미지 저장 디렉토리
SAVE_DIR = PROJECT_ROOT / 'results' / 'realtime_demo_FP'
SAVE_DIR.mkdir(parents=True, exist_ok=True)              # 폴더 없으면 자동 생성

# Threshold                                              # 탐지 신뢰도 임계값
CONF_THRESHOLD = 0.10

# 녹화 여부 (True면 mp4로 저장)                            # 녹화 On/Off
RECORD = True

# 최대 실행 시간 (초, None이면 q키로만 종료)                  # 실행 시간 제한
MAX_SECONDS = 120

# ★ 연속 프레임 수 — 이 값 이상 연속 탐지해야 알람 발생        # 여기서 연속 프레임 수 조절 가능
CONSECUTIVE_FRAMES = 5

# ★ 이미지 저장 간격 (초) — 탐지 중 이 간격마다 저장           # 여기서 저장 간격(초) 조절 가능
SAVE_INTERVAL_SEC = 3

# 지역 좌표 (서해안고속도로)                                # 서해안고속도로 좌표
MIN_X = '126.5'                                          # 최소 경도
MAX_X = '127.0'                                          # 최대 경도
MIN_Y = '37.2'                                           # 최소 위도
MAX_Y = '37.6'                                           # 최대 위도

# ITS API URL                                            # ITS CCTV 정보 API 엔드포인트
ITS_API_URL = "https://openapi.its.go.kr:9443/cctvInfo"

# =============================================================================
# CCTV 목록 조회
# =============================================================================

def get_cctv_list(api_key, min_x, max_x, min_y, max_y, road_type='its'):
    """ITS API로 CCTV 목록 조회 (its=국도, ex=고속도로)"""  # 함수 설명 docstring
    params = {                                           # API 요청 파라미터 딕셔너리
        'apiKey': api_key,                               # 인증 키
        'type': road_type,                               # 도로 유형 (its/ex)
        'cctvType': 1,                                   # 1=실시간 스트리밍
        'minX': min_x,                                   # 최소 경도
        'maxX': max_x,                                   # 최대 경도
        'minY': min_y,                                   # 최소 위도
        'maxY': max_y,                                   # 최대 위도
        'getType': 'json'                                # 응답 형식 JSON
    }
    try:                                                 # 예외 처리 시작
        response = requests.get(                         # GET 요청 보내기
            ITS_API_URL,                                 # API URL
            params=params,                               # 쿼리 파라미터
            timeout=10                                   # 타임아웃 10초
        )
        data = response.json()                           # JSON 파싱
        if 'response' in data and 'data' in data['response']:  # 데이터 존재 여부 확인
            return data['response']['data']              # CCTV 리스트 반환
        return []                                        # 데이터 없으면 빈 리스트
    except Exception as e:                               # 예외 발생 시
        print(f"❌ API 호출 실패: {e}")                    # 에러 메시지 출력
        return []                                        # 빈 리스트 반환

# =============================================================================
# ★ 이미지 저장 헬퍼 함수
# =============================================================================

def save_detection_images(original_frame, bbox_frame, save_dir):
    """탐지 시 원본 이미지와 bbox 이미지를 각각 저장"""      # 함수 설명 docstring
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # 현재 시각을 문자열로 변환
    orig_path = save_dir / f'{timestamp}_원본.jpg'        # 원본 이미지 파일 경로
    bbox_path = save_dir / f'{timestamp}_bbox.jpg'       # bbox 이미지 파일 경로
    cv2.imwrite(str(orig_path), original_frame)          # 원본 이미지 저장
    cv2.imwrite(str(bbox_path), bbox_frame)              # bbox 이미지 저장
    print(f"📸 이미지 저장: {orig_path.name}, {bbox_path.name}")  # 저장 결과 출력
    return orig_path, bbox_path                          # 저장 경로 반환

# =============================================================================
# 실시간 탐지 함수 (후처리 로직 + 이미지 저장 포함)
# =============================================================================

def run_realtime_detection(model, stream_url, cctv_name,
                           conf_threshold=0.10,          # 신뢰도 임계값
                           record=False,                 # 녹화 여부
                           record_path=None,             # 녹화 파일 경로
                           max_seconds=None,             # 최대 실행 시간
                           consecutive_threshold=5,      # 연속 프레임 기준
                           save_interval=3,              # 이미지 저장 간격(초)
                           save_dir=None):               # 이미지 저장 디렉토리
    """실시간 CCTV 화재 탐지 (후처리 + 이미지 저장)"""       # 함수 설명 docstring

    cap = cv2.VideoCapture(stream_url)                   # 스트림 연결

    if not cap.isOpened():                               # 연결 실패 시
        print("❌ 스트림 연결 실패")                        # 에러 메시지 출력
        return None                                      # None 반환

    # 영상 정보                                           # 영상 너비/높이 가져오기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 프레임 너비
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 프레임 높이

    print(f"\n{'='*60}")                                  # 구분선 출력
    print(f"🔴 실시간 화재 탐지 시작")                      # 시작 메시지
    print(f"   CCTV: {cctv_name}")                       # CCTV 이름 출력
    print(f"   해상도: {w}x{h}")                           # 해상도 출력
    print(f"   Threshold: {conf_threshold}")             # 임계값 출력
    print(f"   연속 프레임 기준: {consecutive_threshold}프레임")  # 연속 프레임 기준 출력
    print(f"   이미지 저장 간격: {save_interval}초")        # 저장 간격 출력
    if record:                                           # 녹화 시
        print(f"   녹화: {record_path}")                   # 녹화 경로 출력
    print(f"   종료: 'q' 키")                              # 종료 안내
    print(f"{'='*60}")                                    # 구분선 출력

    # 창 생성                                             # OpenCV 윈도우 생성
    cv2.namedWindow('Fire Detection - CCTV',             # 윈도우 이름
                    cv2.WINDOW_NORMAL)                    # 크기 조절 가능
    cv2.resizeWindow('Fire Detection - CCTV',            # 윈도우 크기 설정
                     min(w, 1280),                       # 최대 너비 1280
                     min(h, 720))                        # 최대 높이 720

    # 녹화 설정                                           # 녹화 초기화
    writer = None                                        # 비디오 라이터 초기값
    if record and record_path:                           # 녹화 설정 시
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')         # 코덱 설정
        writer = cv2.VideoWriter(                        # 비디오 라이터 생성
            str(record_path), fourcc, 15, (w, h))        # 경로, 코덱, FPS, 크기

    # ★ 후처리 로직 변수                                   # 연속 탐지 카운터 초기화
    consecutive_count = 0                                # 연속 탐지 프레임 수
    alarm_active = False                                 # 알람 활성 상태 (False=비활성)

    # ★ 이미지 저장 변수                                   # 이미지 저장 타이밍 초기화
    last_save_time = 0                                   # 마지막 저장 시각 (0=아직 저장 안 함)
    saved_image_count = 0                                # 저장된 이미지 수

    # 통계                                               # 통계 변수 초기화
    total_frames = 0                                     # 총 프레임 수
    detected_frames = 0                                  # 탐지된 프레임 수
    alarm_triggered_count = 0                            # 알람 발생 횟수
    fps_list = []                                        # FPS 기록 리스트
    start_time_total = time.time()                       # 전체 시작 시각
    fire_detected_ever = False                           # 화재 탐지 여부 (전체)

    while True:                                          # 메인 루프 시작
        # 시간 제한                                       # 최대 실행 시간 확인
        if max_seconds and (time.time() - start_time_total) > max_seconds:  # 시간 초과 시
            print(f"\n⏱️ {max_seconds}초 경과 — 자동 종료")  # 종료 메시지
            break                                        # 루프 종료

        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(stream_url)
            continue              # print 없이 조용히 재연결                               # 다음 루프로

        # ★ 원본 프레임 복사 (이미지 저장용)                  # 원본 보존
        original_frame = frame.copy()                    # 원본 프레임 깊은 복사

        # 추론                                           # YOLO 모델 추론
        t_start = time.time()                            # 추론 시작 시각
        results = model.predict(                         # 모델 예측 실행
            frame,                                       # 입력 프레임
            conf=conf_threshold,                         # 신뢰도 임계값
            imgsz=640,                                   # 입력 이미지 크기
            device=0,                                    # GPU 사용 (RTX 4060 Ti)
            save=False,                                  # 결과 저장 안 함
            verbose=False                                # 로그 출력 안 함
        )
        t_end = time.time()                              # 추론 종료 시각

        infer_ms = (t_end - t_start) * 1000              # 추론 시간 (밀리초)
        current_fps = 1000 / infer_ms if infer_ms > 0 else 0  # 현재 FPS 계산
        fps_list.append(current_fps)                     # FPS 리스트에 추가

        # 탐지 결과                                       # 바운딩 박스 수 확인
        num_boxes = len(results[0].boxes)                # 탐지된 객체 수
        frame_detected = num_boxes > 0                   # 이 프레임에서 탐지 여부

        # ★ 후처리 로직: 연속 프레임 카운팅                    # 연속 탐지 판단
        if frame_detected:                               # 이 프레임에서 탐지됐으면
            consecutive_count += 1                       # 연속 카운터 +1
            detected_frames += 1                         # 전체 탐지 프레임 수 +1
        else:                                            # 탐지 안 됐으면
            consecutive_count = 0                        # 연속 카운터 리셋
            alarm_active = False                         # 알람 비활성화

        # ★ 연속 N프레임 이상 탐지 시 알람 발생               # 알람 판단 로직
        if consecutive_count >= consecutive_threshold:   # 연속 카운터 >= 기준값
            if not alarm_active:                         # 알람이 아직 비활성이면
                alarm_active = True                      # 알람 활성화
                alarm_triggered_count += 1               # 알람 발생 횟수 +1
                fire_detected_ever = True                # 전체 화재 탐지 플래그 True
                print(f"🚨 알람 발생! 연속 {consecutive_count}프레임 탐지")  # 알람 메시지
                # ★ 탐지 시작 시점에 즉시 이미지 저장          # 첫 알람 시 즉시 저장
                if save_dir:                             # 저장 경로가 있으면
                    bbox_frame = results[0].plot()       # bbox가 그려진 프레임 생성
                    save_detection_images(               # 이미지 저장 함수 호출
                        original_frame,                  # 원본 프레임
                        bbox_frame,                      # bbox 프레임
                        save_dir                         # 저장 디렉토리
                    )
                    saved_image_count += 1               # 저장 카운트 +1
                    last_save_time = time.time()         # 마지막 저장 시각 갱신

            # ★ 알람 활성 중: 저장 간격마다 이미지 저장         # 주기적 저장 로직
            elif save_dir and (time.time() - last_save_time) >= save_interval:  # 간격 경과 시
                bbox_frame = results[0].plot()           # bbox가 그려진 프레임 생성
                save_detection_images(                   # 이미지 저장 함수 호출
                    original_frame,                      # 원본 프레임
                    bbox_frame,                          # bbox 프레임
                    save_dir                             # 저장 디렉토리
                )
                saved_image_count += 1                   # 저장 카운트 +1
                last_save_time = time.time()             # 마지막 저장 시각 갱신

        # 화면 표시용 프레임 생성                            # 디스플레이 프레임 결정
        if frame_detected:                               # 탐지됐으면
            display_frame = results[0].plot()            # bbox 포함 프레임
        else:                                            # 탐지 안 됐으면
            display_frame = frame.copy()                 # 원본 프레임 복사

        # 상단 정보 바                                     # 상태바 색상 결정
        if alarm_active:                                 # 알람 활성 시
            bar_color = (0, 0, 255)                      # 빨간색 (BGR)
        elif frame_detected:                             # 탐지만 됐을 때 (알람 전)
            bar_color = (0, 140, 255)                    # 주황색 (BGR)
        else:                                            # 정상 상태
            bar_color = (50, 50, 50)                     # 회색 (BGR)

        cv2.rectangle(display_frame,                     # 상단 바 그리기
                      (0, 0), (w, 55),                   # 좌상단~우측 55px
                      bar_color, -1)                     # 색상, 채우기

        # CCTV 이름                                       # 좌상단에 CCTV명 표시
        cv2.putText(display_frame,                       # 텍스트 그리기
                    f"CCTV: {cctv_name[:30]}",           # CCTV 이름 (30자 제한)
                    (10, 18),                            # 위치 (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,            # 폰트
                    0.5, (255, 255, 255), 1)              # 크기, 흰색, 두께

        # FPS                                             # 우상단에 FPS 표시
        cv2.putText(display_frame,                       # 텍스트 그리기
                    f"FPS: {current_fps:.1f}",           # FPS 값
                    (w - 120, 18),                       # 위치
                    cv2.FONT_HERSHEY_SIMPLEX,            # 폰트
                    0.5, (255, 255, 255), 1)              # 크기, 흰색, 두께

        # ★ 탐지 상태 (후처리 반영)                         # 상태 메시지 결정
        if alarm_active:                                 # 알람 활성 시
            status = f"🚨 FIRE ALARM! (연속 {consecutive_count}프레임)"  # 알람 메시지
            status_color = (0, 0, 255)                   # 빨간색
        elif frame_detected:                             # 탐지 중 (알람 전)
            status = f"Detecting... ({consecutive_count}/{consecutive_threshold})"  # 카운팅 중
            status_color = (0, 200, 255)                 # 노란색
        else:                                            # 정상 상태
            status = "Monitoring..."                     # 모니터링 중
            status_color = (200, 200, 200)               # 밝은 회색

        cv2.putText(display_frame, status,               # 상태 텍스트 그리기
                    (10, 38),                            # 위치
                    cv2.FONT_HERSHEY_SIMPLEX,            # 폰트
                    0.6, status_color, 2)                 # 크기, 색상, 두께

        # Threshold + 연속 프레임 정보                      # 추가 정보 표시
        info_text = f"Thr:{conf_threshold} | Consec:{consecutive_count}/{consecutive_threshold}"  # 정보 문자열
        cv2.putText(display_frame, info_text,            # 텍스트 그리기
                    (w - 350, 38),                       # 위치
                    cv2.FONT_HERSHEY_SIMPLEX,            # 폰트
                    0.4, (200, 200, 200), 1)              # 크기, 색상, 두께

        # ★ 저장 상태 표시                                 # 저장 카운트 표시
        cv2.putText(display_frame,                       # 텍스트 그리기
                    f"Saved: {saved_image_count}",       # 저장된 이미지 수
                    (10, 52),                            # 위치
                    cv2.FONT_HERSHEY_SIMPLEX,            # 폰트
                    0.35, (180, 255, 180), 1)             # 크기, 연두색, 두께

        # 녹화 표시                                       # 녹화 인디케이터
        if record:                                       # 녹화 중이면
            cv2.circle(display_frame,                    # 빨간 원 그리기
                       (w - 20, 12), 6,                  # 위치, 반지름
                       (0, 0, 255), -1)                  # 빨간색, 채우기
            cv2.putText(display_frame, "REC",            # REC 텍스트
                        (w - 55, 16),                    # 위치
                        cv2.FONT_HERSHEY_SIMPLEX,        # 폰트
                        0.4, (0, 0, 255), 1)              # 크기, 빨간색, 두께

        # 화면 출력                                       # 프레임 디스플레이
        cv2.imshow('Fire Detection - CCTV',              # 윈도우에 프레임 출력
                   display_frame)

        # 녹화                                           # 비디오 파일에 프레임 쓰기
        if writer:                                       # 비디오 라이터가 있으면
            writer.write(display_frame)                  # 프레임 저장

        total_frames += 1                                # 총 프레임 수 +1

        # 'q' 키 종료                                     # 키 입력 확인
        key = cv2.waitKey(1) & 0xFF                      # 1ms 대기, 키 값 마스킹
        if key == ord('q'):                              # q 키 입력 시
            print("\n🛑 'q' 키로 종료")                    # 종료 메시지
            break                                        # 루프 종료

    # 정리                                               # 리소스 해제
    cap.release()                                        # 카메라 해제
    if writer:                                           # 비디오 라이터가 있으면
        writer.release()                                 # 라이터 해제
    cv2.destroyAllWindows()                              # 모든 창 닫기

    # 통계                                               # 실행 결과 통계 계산
    elapsed = time.time() - start_time_total             # 총 경과 시간
    avg_fps = np.mean(fps_list) if fps_list else 0       # 평균 FPS 계산

    stats = {                                            # 통계 딕셔너리
        'cctv_name': cctv_name,                          # CCTV 이름
        'total_frames': total_frames,                    # 총 프레임 수
        'detected_frames': detected_frames,              # 탐지 프레임 수
        'detection_rate': round(                         # 탐지율 계산
            detected_frames / total_frames * 100, 1      # 퍼센트 변환
        ) if total_frames > 0 else 0,                    # 0 프레임 방지
        'alarm_triggered_count': alarm_triggered_count,  # 알람 발생 횟수
        'saved_image_count': saved_image_count,          # 저장된 이미지 수
        'avg_fps': round(avg_fps, 1),                    # 평균 FPS
        'elapsed_seconds': round(elapsed, 1),            # 경과 시간
        'fire_detected': fire_detected_ever              # 화재 탐지 여부
    }

    print(f"\n📊 실행 통계:")                              # 통계 제목
    print(f"   총 프레임: {total_frames}")                  # 총 프레임 수
    print(f"   탐지 프레임: {detected_frames} ({stats['detection_rate']}%)")  # 탐지율
    print(f"   알람 발생: {alarm_triggered_count}회")       # 알람 횟수
    print(f"   저장 이미지: {saved_image_count}세트")        # 저장 이미지 수
    print(f"   평균 FPS: {avg_fps:.1f}")                   # 평균 FPS
    print(f"   실행 시간: {elapsed:.1f}초")                 # 경과 시간
    print(f"   화재 탐지: {'🔥 있음' if fire_detected_ever else '✅ 없음 (정상)'}")  # 탐지 결과

    return stats                                         # 통계 딕셔너리 반환

# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':                               # 스크립트 직접 실행 시
    print("=" * 60)                                      # 구분선
    print("🔥 공공 CCTV 실시간 화재 탐지 데모 (v2)")        # 제목 출력
    print("   — 후처리 로직 + 이미지 자동 저장")             # 부제 출력
    print("=" * 60)                                      # 구분선

    # 1. 모델 로드                                        # YOLO 모델 로드
    print("\n📦 모델 로드 중...")                           # 로드 중 메시지
    model = YOLO(str(MODEL_PATH))                        # YOLO 모델 생성
    print(f"✅ 모델 로드 완료: {MODEL_PATH.name}")          # 완료 메시지

    # 2. CCTV 목록 조회                                    # API로 CCTV 목록 가져오기
    print("\n📡 CCTV 목록 조회 중...")                      # 조회 중 메시지
    its_cctvs = get_cctv_list(                           # 국도 CCTV 조회
        API_KEY, MIN_X, MAX_X, MIN_Y, MAX_Y, 'its')
    ex_cctvs = get_cctv_list(                            # 고속도로 CCTV 조회
        API_KEY, MIN_X, MAX_X, MIN_Y, MAX_Y, 'ex')
    all_cctvs = its_cctvs + ex_cctvs                     # 전체 CCTV 합치기
    print(f"총 {len(all_cctvs)}개 CCTV 발견")              # 발견 개수 출력

    if len(all_cctvs) == 0:                              # CCTV 없으면
        print("❌ CCTV를 찾지 못했습니다. API 키와 좌표를 확인하세요.")  # 에러 메시지
        exit()                                           # 프로그램 종료

    # 3. CCTV 목록 출력                                    # CCTV 리스트 화면 출력
    print(f"\n{'번호':<6} {'CCTV명':<40}")                 # 헤더 출력
    print("-" * 50)                                      # 구분선
    for i, cctv in enumerate(all_cctvs[:30]):            # 최대 30개까지 출력
        name = cctv.get('cctvname', '이름없음')[:38]      # CCTV 이름 (38자 제한)
        print(f"{i:<6} {name}")                          # 번호와 이름 출력
    if len(all_cctvs) > 30:                              # 30개 초과 시
        print(f"... 외 {len(all_cctvs) - 30}개")          # 나머지 개수 표시

    # 4. CCTV 선택                                        # 사용자 CCTV 선택
    while True:                                          # 올바른 입력까지 반복
        try:                                             # 예외 처리
            idx = int(input(                             # 번호 입력받기
                f"\n▶ CCTV 번호를 입력하세요 (0~{len(all_cctvs)-1}): "))
            if 0 <= idx < len(all_cctvs):                # 범위 내 확인
                break                                    # 루프 종료
            print("범위를 벗어났습니다.")                    # 범위 초과 메시지
        except ValueError:                               # 숫자 아닌 입력
            print("숫자를 입력하세요.")                      # 안내 메시지

    selected = all_cctvs[idx]                            # 선택된 CCTV 정보
    cctv_name = selected.get('cctvname', '이름없음')      # CCTV 이름
    cctv_url = selected.get('cctvurl', '')               # CCTV 스트림 URL

    print(f"\n선택: {cctv_name}")                         # 선택 결과 출력

    # 5. 연결 테스트                                       # 스트림 연결 테스트
    print("🔌 연결 테스트 중...")                            # 테스트 메시지
    test_cap = cv2.VideoCapture(cctv_url)                # 테스트 연결
    if not test_cap.isOpened():                          # 연결 실패 시
        print("❌ 연결 실패 — 다른 CCTV를 선택해주세요")      # 실패 메시지
        test_cap.release()                               # 연결 해제
        exit()                                           # 프로그램 종료
    ret, _ = test_cap.read()                             # 프레임 읽기 테스트
    test_cap.release()                                   # 테스트 연결 해제
    if not ret:                                          # 프레임 읽기 실패 시
        print("❌ 프레임 읽기 실패 — 다른 CCTV를 선택해주세요")  # 실패 메시지
        exit()                                           # 프로그램 종료
    print("✅ 연결 성공!")                                  # 성공 메시지

    # 6. 녹화 파일 경로                                    # 녹화 파일 경로 설정
    record_path = None                                   # 초기값 None
    if RECORD:                                           # 녹화 설정 시
        timestamp = datetime.now().strftime(             # 현재 시각 문자열
            '%Y%m%d_%H%M%S')
        safe_name = cctv_name.replace(                   # 안전한 파일명 생성
            ' ', '_').replace('/', '_')[:20]
        record_path = DEMO_DIR / f'demo_{timestamp}_{safe_name}.mp4'  # 녹화 파일 경로

    # 7. 실시간 탐지 실행                                   # 메인 탐지 실행
    stats = run_realtime_detection(                      # 탐지 함수 호출
        model=model,                                     # YOLO 모델
        stream_url=cctv_url,                             # 스트림 URL
        cctv_name=cctv_name,                             # CCTV 이름
        conf_threshold=CONF_THRESHOLD,                   # 신뢰도 임계값
        record=RECORD,                                   # 녹화 여부
        record_path=record_path,                         # 녹화 경로
        max_seconds=120,                                # 최대 실행 시간
        consecutive_threshold=CONSECUTIVE_FRAMES,        # ★ 연속 프레임 기준
        save_interval=SAVE_INTERVAL_SEC,                 # ★ 이미지 저장 간격
        save_dir=SAVE_DIR                                # ★ 이미지 저장 경로
    )

    # 8. 결과 저장                                        # JSON 결과 저장
    if stats:                                            # 통계가 있으면
        result_data = {                                  # 결과 데이터 딕셔너리
            "test_date": datetime.now().strftime(        # 테스트 날짜
                "%Y-%m-%d %H:%M:%S"),
            "model": "YOLOv8n_tuned",                    # 모델 이름
            "threshold": CONF_THRESHOLD,                 # 임계값
            "consecutive_frames": CONSECUTIVE_FRAMES,    # ★ 연속 프레임 기준
            "save_interval_sec": SAVE_INTERVAL_SEC,      # ★ 저장 간격
            "cctv_name": cctv_name,                      # CCTV 이름
            "stats": stats                               # 통계 데이터
        }
        json_path = DEMO_DIR / 'realtime_demo_results.json'  # JSON 파일 경로
        with open(json_path, 'w', encoding='utf-8') as f:    # 파일 열기
            json.dump(result_data, f,                    # JSON 저장
                      indent=2, ensure_ascii=False)      # 들여쓰기, 유니코드
        print(f"\n💾 결과 저장: {json_path}")               # 저장 경로 출력
        if RECORD and record_path and record_path.exists():  # 녹화 파일 확인
            size_mb = record_path.stat().st_size / (     # 파일 크기 계산
                1024 * 1024)                             # MB 변환
            print(f"💾 녹화 파일: {record_path} ({size_mb:.1f} MB)")  # 크기 출력

    print("\n✅ 데모 종료!")                                # 종료 메시지
