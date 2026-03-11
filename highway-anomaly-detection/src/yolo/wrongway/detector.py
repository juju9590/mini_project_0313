# 메인 오케스트레이터
# 모든 모듈을 조립하고 run() 루프를 실행

import cv2
import numpy as np
import time

from .config import DetectorConfig
from .state import DetectorState
from .flow_map import FlowMap
from .tracker import YoloTracker
from .judge import WrongWayJudge
from .id_manager import IDManager
from .camera_switch import CameraSwitchDetector
from .visualizer import Visualizer
from .logger import CSVLogger
from .bbox_stabilizer import BBoxStabilizer

class Detector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

        # ── 런타임 상태(state) + 모듈 생성 ──────────────────────────────
        # state: 프레임 번호, 궤적, 역주행 카운트/확정, 라벨 매핑 등
        self.state = DetectorState()

        # flow: 정상 차량들의 평균 흐름(방향 벡터)을 그리드로 학습/보간/저장
        self.flow = FlowMap(cfg.grid_size, cfg.alpha, cfg.min_samples)

        # tracker: YOLO 검출 + ByteTrack 추적 → id, bbox, center 반환
        self.tracker = YoloTracker(cfg.model_path, cfg.conf, cfg.target_classes)

        # judge: flow_map과 차량 이동 방향의 코사인 유사도 + 투표/히스테리시스로 역주행 판정
        self.judge = WrongWayJudge(cfg, self.flow, self.state)

        # idm: W1/W2 라벨 관리 + occlusion 후 재등장 재매칭 + 오래된 트랙 정리
        self.idm = IDManager(cfg, self.flow, self.state)

        # switch: 장면/카메라 전환 감지 (160x90 grayscale diff 기반)
        self.switch = CameraSwitchDetector(cfg)

        # vis: 시각화(박스/궤적/패널/디버그 표시)
        self.vis = Visualizer(cfg, self.state, self.flow)

        # logger: 프레임/트랙/이벤트 단위 CSV 로그 저장 (cfg.log_dir 지정 시만 활성)
        self.logger = CSVLogger(cfg.log_dir) if cfg.log_dir else None

        # bbox_stab: 바운딩박스 좌표 EMA 안정화기
        # ### 목적: bbox jitter로 인해 정지/저속 차량도 "가짜 이동"이 생기는 문제 완화
        # ### 효과: 중심점(cx, cy) 궤적이 부드러워져 speed/ndx/ndy 계산이 안정됨
        self.bbox_stab = BBoxStabilizer(alpha=0.5)

        # ---------------- flow_map 로드 (탐지 전용이라면 필수) ----------------
        if cfg.flow_map_path:
            loaded = self.flow.load(cfg.flow_map_path)

            # 탐지 전용 모드인데 flow_map이 없으면 그냥 돌리면 전부 None/0이 되어 의미가 없어짐
            # → 잘못된 실험을 방지하기 위해 즉시 예외로 종료
            if not loaded and cfg.detect_only:
                raise FileNotFoundError(f"detect_only=True 인데 flow_map이 없습니다: {cfg.flow_map_path}")

            # detect_only가 아니면 flow_map 없을 때 학습 모드로 들어갈 수도 있음(선택)
            if not loaded and (not cfg.detect_only):
                self.state.is_learning = True
                print("flow_map 없음 → 학습 모드 시작")

        else:
            # detect_only인데 flow_map_path 자체가 None이면 설정 오류
            if cfg.detect_only:
                raise ValueError("detect_only=True 인데 flow_map_path가 None 입니다.")

        # ── 결과 저장 폴더 ──
        if cfg.result_dir:
            cfg.result_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 기본 유틸 ====================
    def _get_next_filename(self, base="results", ext=".mp4"):
        """결과 파일명이 겹치지 않도록 뒤에 번호를 붙여서 새 파일명 생성"""
        idx = 1
        while True:
            p = self.cfg.result_dir / f"{base}_{idx}{ext}"  # 예: results_1.mp4
            if not p.exists():   # 해당 파일이 아직 존재하지 않으면
                return p         # 이 이름을 사용
            idx += 1             # 존재하면 번호 증가 후 다시 검사

    # ==================== 메인 루프 ====================
    def run(self, video_name):
        """영상 파일을 열어 프레임 단위로 처리하며 역주행을 감지/표시/저장"""
        cfg = self.cfg
        st = self.state

        video_path = cfg.data_dir / video_name   # 입력 비디오 경로
        if not video_path.exists():
            print(f"파일 없음: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))   # 비디오 캡처 객체 생성
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 프레임 너비
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 프레임 높이
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0        # 원본 FPS (없으면 30)

        st.frame_w, st.frame_h, st.video_fps = fw, fh, fps  # state에 저장
        self.flow.init_grid(fw, fh)                          # 그리드 초기화

        save_path = self._get_next_filename()                    # 결과 저장 파일명
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")                 # mp4 인코더 설정
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (fw, fh))
        print(f"📹 저장: {save_path}")

        prev_time = time.time()  # FPS 계산 시작 시간

        # 키보드 단축키 안내
        print("\n" + "=" * 50)
        print("  [T] 궤적  [D] 방향  [F] 흐름장")
        print("  [S] 속도  [I] 패널  [V] 투표 디버그")
        print("  [P] 내적값  [Q] 종료")
        print("=" * 50 + "\n")

        while cap.isOpened():                    # 비디오 스트림이 열려 있는 동안
            ret, frame = cap.read()              # 프레임 읽기
            if not ret:
                break                            # 더 이상 프레임 없으면 종료

            st.frame_num += 1                    # 프레임 번호 증가

            # ── YOLO 추적 ──
            tracks = self.tracker.track(frame)   # [{id, x1, y1, x2, y2, cx, cy}, ...]
            active_ids = {t["id"] for t in tracks}  # 현재 프레임에 보이는 ID들

            # 처음 등장한 프레임 기록
            for t in tracks:
                if t["id"] not in st.first_seen_frame:
                    st.first_seen_frame[t["id"]] = st.frame_num
                    
            # 현재 프레임 시간(초) = frame / fps
            time_sec = st.frame_num / st.video_fps

            # 역주행 확정 차량 수 (W라벨 기준으로 중복 제거)
            wrong_confirmed_count = len(set(st.display_id_map.values()))
            
            # 이번 프레임에 로그를 남길지 여부 (5프레임마다)
            log_this_frame = (self.logger is not None) and (st.frame_num % cfg.log_interval_frames == 0)

            # 영상 타임라인 기준 시간(초)
            time_sec = st.frame_num / st.video_fps
            
            # 프레임 단위 요약 로그 저장
            if log_this_frame:
                wrong_confirmed_count = len(set(st.display_id_map.values()))
                self.logger.log_frame(
                    frame_num=st.frame_num,
                    time_sec=time_sec,
                    active_tracks=len(tracks),
                    wrong_confirmed_count=wrong_confirmed_count,
                    flow_samples_total=self.flow.count.sum(),
                    camera_switch_triggered=False,  # detect_only 실험이면 보통 전환 리셋 비활성로 운용(원하면 실제값 넣기)
                    mode="DETECTING"
                )

            # ── 카메라 전환 감지 ──
            if not st.is_learning and not st.relearning:       # 학습/재학습 중이 아닐 때만
                if self.switch.check(frame, st.frame_num, st.cooldown_until):  # 전환 감지되면
                    st.reset_for_relearn()                     # 상태 초기화
                    self.flow.reset()                          # flow_map 초기화

            # ── 초기 학습 완료 처리 ──
            if st.is_learning and st.frame_num >= cfg.learning_frames:
                self.flow.apply_spatial_smoothing()  # 공간 보정
                if cfg.flow_map_path:
                    self.flow.save(cfg.flow_map_path)  # flow_map 저장
                st.is_learning = False                 # 학습 모드 종료
                print("학습 완료!")

            # ── 재학습 모드 처리 ──
            if st.relearning:
                elapsed = st.frame_num - st.relearn_start_frame
                if elapsed >= cfg.relearn_frames:                    # 재학습 프레임 수 도달 시
                    self.flow.apply_spatial_smoothing()               # 공간 보정
                    if cfg.flow_map_path:
                        self.flow.save(cfg.flow_map_path)            # flow_map 저장
                    st.relearning = False                             # 재학습 모드 종료
                    st.cooldown_until = st.frame_num + cfg.cooldown_frames  # 쿨다운 설정
                    self.switch.set_reference(frame)                 # 새 기준 프레임 설정
                    print("재학습 완료! 쿨다운 시작")

            # ── 차량별 처리 ──
            for t in tracks:
                tid = t["id"]
                # 바운딩박스 EMA 안정화 적용
                # raw_bbox: YOLO가 낸 원본 bbox (프레임마다 떨림 가능)
                raw_bbox = (t["x1"], t["y1"], t["x2"], t["y2"])
                # stabilize 결과:
                #  - (x1,y1,x2,y2): 안정화된 bbox(정수)
                #  - (cx,cy): 안정화된 중심점(float)
                # 이후 로직은 이 중심점(cx,cy)을 궤적/속도 계산에 사용 → jitter 영향 감소
                x1, y1, x2, y2, cx, cy = self.bbox_stab.stabilize(
                    tid, raw_bbox, st.frame_num
                )
                
                # ID 재매칭 시도 (학습 모드가 아닐 때만)
                if not st.is_learning and not st.relearning:
                    self.idm.check_reappear(tid, cx, cy)

                # 궤적에 현재 위치 추가
                st.trajectories[tid].append((cx, cy))
                if len(st.trajectories[tid]) > cfg.trail_length:
                    st.trajectories[tid].pop(0)  # 오래된 궤적 제거

                traj = st.trajectories[tid]      # 궤적 참조
                is_wrong = False                 # 이번 프레임에서 역주행 여부
                speed = 0                        # 속도
                ndx, ndy = 0, 0                  # 단위 방향 벡터
                debug_info = {}                  # 디버그 정보

                # 궤적 길이가 velocity_window 이상일 때만 속도/방향 계산
                if len(traj) >= cfg.velocity_window:
                    vdx = traj[-1][0] - traj[-cfg.velocity_window][0]  # x 이동량
                    vdy = traj[-1][1] - traj[-cfg.velocity_window][1]  # y 이동량
                    mag = np.sqrt(vdx ** 2 + vdy ** 2)                  # 속도 크기

                    # 프레임당 평균 이동거리 계산 (떨림 필터)
                    avg_move = mag / cfg.velocity_window

                    # 누적 이동거리 AND 프레임당 이동거리 모두 충족해야 "움직이는 중"
                    if mag > cfg.min_move_distance and avg_move > cfg.min_move_per_frame:
                        ndx, ndy = vdx / mag, vdy / mag
                        speed = mag                          # 속도

                        if st.is_learning or st.relearning:
                            # 학습 모드: 흐름장 업데이트
                            self.flow.learn_step(
                                traj[-cfg.velocity_window][0],
                                traj[-cfg.velocity_window][1],
                                cx, cy, cfg.min_move_distance
                            )
                        else:
                            # 감지 모드: 역주행 여부 판단
                            is_wrong, _, debug_info = self.judge.check(
                                tid, traj, ndx, ndy, mag, cy
                            )
                            
                            # 역주행 의심이 전혀 없으면 정상 흐름으로 학습에 사용
                            # 감지만할 때 주석 처리하면 flow_map 완전히 고정
                            if (cfg.enable_online_flow_update and
                                (not is_wrong) and
                                (st.wrong_way_count[tid] == 0)):

                                self.flow.learn_step(
                                    traj[-cfg.velocity_window][0],
                                    traj[-cfg.velocity_window][1],
                                    cx, cy,
                                    cfg.min_move_distance
                                )
                                
                            # 역주행 확정 시 라벨 부여
                            if is_wrong and tid in st.wrong_way_ids:
                                self.idm.assign_label(tid)
                else:
                    # 궤적이 짧더라도 이미 역주행 확정된 차량이면 그대로 표시
                    if tid in st.wrong_way_ids:
                        is_wrong = True
                        debug_info = {"status": "CONFIRMED", "cos_values": []}
                                        
                # ---------------- 트랙 단위(프레임 × 트랙) 로그 생성 ----------------
                # 현재 프레임에서 "그 차량이 왜 역주행으로 의심되었는지"를 숫자로 남기기 위함

                # 원근 기반 속도 임계값(현재 위치 cy 기준) → 느리면 판정 skip되는지 확인 가능
                speed_thr = self.judge.get_speed_threshold(cy)

                # 해당 위치에서 보간된 정상 흐름 벡터와, 현재 이동방향의 cos 값을 저장
                flow_v = None
                cos_current = None

                # speed가 0이면 방향 벡터가 무의미하므로, 움직일 때만 계산
                if speed > 0:
                    flow_v = self.flow.get_interpolated(cx, cy)
                    if flow_v is not None:
                        cos_current = float(ndx * flow_v[0] + ndy * flow_v[1])

                # judge(check)에서 준 debug_info를 정형 컬럼으로 요약해서 저장
                agree = debug_info.get("agree", "")
                disagree = debug_info.get("disagree", "")
                skip = debug_info.get("skip", "")
                total = debug_info.get("total", "")
                judge_status = debug_info.get("status", "")

                # 궤적이 짧아서 아예 judge를 안 탄 경우 상태 표시
                if judge_status == "" and len(traj) < cfg.velocity_window:
                    judge_status = "short"

                # disagree_ratio 계산(가능할 때만)
                disagree_ratio = ""
                if isinstance(total, int) and total > 0 and isinstance(disagree, int):
                    disagree_ratio = round(disagree / total, 4)

                # 사용자 표시 라벨(W1,W2...) 저장(없으면 빈칸)
                display_label = self.idm.get_display_label(tid) or ""

                # 역주행 확정 여부(정형 데이터로 쓰기 좋게 0/1)
                is_confirmed = int(tid in st.wrong_way_ids)

                # 최종 1행 저장
                if log_this_frame:
                    self.logger.log_track([
                        st.frame_num, round(time_sec, 3), tid, display_label,
                        round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                        round(cx, 2), round(cy, 2),
                        round(speed, 3) if speed else "",
                        round(ndx, 5) if speed else "",
                        round(ndy, 5) if speed else "",
                        round(speed_thr, 3) if speed_thr else "",
                        round(float(flow_v[0]), 5) if flow_v is not None else "",
                        round(float(flow_v[1]), 5) if flow_v is not None else "",
                        round(cos_current, 5) if cos_current is not None else "",
                        judge_status, agree, disagree, skip, total,
                        disagree_ratio,
                        st.wrong_way_count.get(tid, 0),
                        is_confirmed
                    ])
                    
                # 시각화용 플래그: 이미 확정된 역주행도 항상 붉게 표시
                is_wrong_display = is_wrong or (tid in st.wrong_way_ids)

                if is_wrong_display:
                    # 역주행 차량의 현재 위치 저장 (ID 재매칭용)
                    st.wrong_way_last_pos[tid] = (cx, cy, st.frame_num)

                # ── 시각화 ──
                if self.vis.show_trails:
                    self.vis.draw_trajectory(frame, tid, is_wrong_display)

                if self.vis.show_direction and speed > 3:
                    self.vis.draw_direction_arrow(frame, cx, cy, ndx, ndy,
                                                 speed, is_wrong_display)

                if is_wrong_display:
                    self.vis.draw_wrong_way_alert(frame, tid, x1, y1, x2, y2)
                else:
                    # 정상 차량은 초록 박스/ID만 표시
                    self.vis.draw_normal_box(frame, tid, x1, y1, x2, y2)

                if self.vis.show_speed and speed > 3:
                    self.vis.draw_speed_label(frame, x1, y2, speed, cy, is_wrong_display)

                if self.vis.show_vote_debug and debug_info:
                    self.vis.draw_vote_debug(frame, tid, debug_info, x2, y1)

                # 내적값(코사인) 히스토리 저장/표시
                cos_vals = debug_info.get("cos_values", [])
                if cos_vals:
                    st.last_cos_values[tid] = cos_vals

                if self.vis.show_dot_product and st.last_cos_values.get(tid):
                    self.vis.draw_dot_product(frame, x2, y1,
                                             st.last_cos_values[tid],
                                             is_wrong_display)

            # ── 트랙 정리 ──
            # IDManager: trajectories/last_pos 등 정리
            # BBoxStabilizer: smoothed bbox 캐시 정리 (안 하면 사라진 ID가 계속 남음)
            if st.frame_num % 30 == 0:          # 30프레임마다
                self.idm.cleanup(active_ids)
                self.bbox_stab.cleanup(active_ids)

            # 흐름장(배경 화살표) 표시
            if self.vis.show_flow:
                self.vis.draw_flow(frame)

            # 정보 패널 표시
            if self.vis.show_info_panel:
                self.vis.draw_info_panel(frame, len(tracks))

            # 탐지 소요시간 통계 패널
            if self.vis.show_detection_stats:
                self.vis.draw_detection_stats(frame)

            # 재학습 중이면 화면 상단에 상태 텍스트 표시
            if st.relearning:
                elapsed = st.frame_num - st.relearn_start_frame
                progress = min(100, elapsed / cfg.relearn_frames * 100)
                cv2.putText(frame, f"CAMERA SWITCHED - RE-LEARNING: {progress:.0f}%",
                            (fw // 2 - 220, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            # FPS 계산 및 표시
            curr_time = time.time()
            show_fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_y = 255 if self.vis.show_info_panel else 30
            cv2.putText(frame, f"FPS: {show_fps:.1f}", (10, fps_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)                                        # 결과 영상 파일에 프레임 기록
            cv2.imshow("Highway Wrong-Way Detection", frame)           # 화면에 출력

            key = cv2.waitKey(1) & 0xFF                                # 키 입력 대기
            if key == ord("q"):
                break                                                  # q 누르면 종료
            self.vis.handle_keys(key)                                  # 시각화 옵션 토글 등 처리

            # 학습/재학습 중에는 주기적으로 공간 보정 적용
            if (st.is_learning or st.relearning) and st.frame_num % 150 == 0:
                self.flow.apply_spatial_smoothing()

        # ── 루프 종료 후 정리 ──
        cap.release()            # 비디오 캡처 해제
        writer.release()         # 비디오 라이터 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

        if not st.is_learning and cfg.flow_map_path:  # 학습이 완료된 상태라면
            self.flow.save(cfg.flow_map_path)          # flow_map 저장

        self._print_final_stats(save_path)
        # ── 이벤트 로그 저장(확정된 W1/W2 통계 덤프) ──
        if self.logger:
            self.logger.log_events_from_stats(st.detection_stats)
            self.logger.close()

    # ==================== 최종 통계 출력 ====================
    def _print_final_stats(self, save_path):
        """최종 탐지 소요시간 통계 콘솔 출력"""
        st = self.state

        # detection_stats에 기록된 역주행 차량(라벨)의 수 = 최종 역주행 차량 수
        total_wrong = len(st.detection_stats)

        print(f"\n✅ 저장 완료: {save_path} ({st.frame_num} 프레임)")
        print(f"   총 역주행 차량: {total_wrong}대")

        # 각 역주행 라벨(W1, W2...)에 어떤 ID들이 쓰였는지 출력
        # (ID 재매칭으로 한 라벨에 여러 ID가 붙을 수 있음)
        for label in sorted(set(st.display_id_map.values())):
            ids = [k for k, v in st.display_id_map.items() if v == label]
            print(f"   {label}: 사용된 ID {ids}")

        # detection_stats가 비어있지 않으면 (역주행이 1대 이상 있으면) 상세 통계 출력
        if st.detection_stats:
            print(f"\n{'=' * 60}")
            print(f" 역주행 탐지 소요시간 통계 (FPS: {st.video_fps:.1f})")
            print(f"{'=' * 60}")
            print(f"  {'차량':>6} │ {'등장→확정':>18} │ {'의심→확정':>18}")
            print(f"  {'─' * 6}─┼─{'─' * 18}─┼─{'─' * 18}")

            all_appear = []   # 각 차량의 '등장→확정' 소요 시간(초)을 모아두는 리스트
            all_suspect = []  # 각 차량의 '의심→확정' 소요 시간(초)을 모아두는 리스트

            # detection_stats: { 'W1': {...}, 'W2': {...}, ... } 형태
            for label, s in sorted(st.detection_stats.items()):
                # 예: "  14f ( 0.47s)" 형식의 문자열 구성
                appear_str = f"{s['frames_from_appear']:>4}f ({s['seconds_from_appear']:>5.2f}s)"
                suspect_str = f"{s['frames_from_suspect']:>4}f ({s['seconds_from_suspect']:>5.2f}s)"

                # 한 줄씩: 차량 라벨, 등장→확정, 의심→확정 시간 출력
                print(f"  {label:>6} │ {appear_str:>18} │ {suspect_str:>18}")

                # 전체 통계(평균/최소/최대)를 위해 초 단위 값만 따로 저장
                all_appear.append(s["seconds_from_appear"])
                all_suspect.append(s["seconds_from_suspect"])

            print(f"  {'─' * 6}─┼─{'─' * 18}─┼─{'─' * 18}")

            # 전체 역주행 차량들에 대한 '등장→확정' / '의심→확정' 시간의
            # 평균, 최소, 최대 값 계산
            avg_a = np.mean(all_appear)
            avg_s = np.mean(all_suspect)
            min_a = np.min(all_appear)
            max_a = np.max(all_appear)
            min_s = np.min(all_suspect)
            max_s = np.max(all_suspect)

            # 요약 통계(평균, 최소, 최대)를 보기 좋게 출력
            print(f"  {'평균':>6} │ {'':>13}{avg_a:>5.2f}s │ {'':>13}{avg_s:>5.2f}s")
            print(f"  {'최소':>6} │ {'':>13}{min_a:>5.2f}s │ {'':>13}{min_s:>5.2f}s")
            print(f"  {'최대':>6} │ {'':>13}{max_a:>5.2f}s │ {'':>13}{max_s:>5.2f}s")
            print(f"{'=' * 60}")
        else:
            # 역주행이 한 번도 감지되지 않은 경우
            print("\n 역주행 차량이 감지되지 않았습니다.")