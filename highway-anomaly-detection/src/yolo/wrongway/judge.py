# 흐름장과 코사인 유사도 기반 역주행 판별
# 원근 기반 속도 게이트 + 다중 포인트 투표 + 시간적 히스테리시스(카운팅)

import numpy as np


class WrongWayJudge:
    def __init__(self, cfg, flow_map, state):
        self.cfg = cfg
        self.flow = flow_map
        self.st = state

    def get_speed_threshold(self, cy):
        """화면 상의 y 위치에 따라 속도 임계값을 달리 적용 (원근 보정)"""
        ratio = cy / self.st.frame_h       # 화면 상단(0)~하단(1) 비율
        scale = 0.3 + 0.7 * ratio          # 위는 0.3배, 아래는 1.0배 근처
        return self.cfg.base_speed_threshold * scale  # 위치에 따른 속도 임계값

    def check(self, track_id, traj, ndx, ndy, speed, cy):
        """
        한 차량에 대해 flow_map과 방향 비교, 투표 방식으로 역주행 여부 판정

        Returns:
            (is_wrong, disagree_ratio, debug_info)
        """
        cfg = self.cfg
        st = self.st

        # 이미 역주행 확정된 차량이면 바로 True
        if track_id in st.wrong_way_ids:
            return True, 1.0, {"status": "CONFIRMED", "cos_values": []}

        # 위치 기반 속도 임계값
        adaptive_threshold = self.get_speed_threshold(cy)
        if speed < adaptive_threshold:  # 너무 느리면 판정 X
            return False, 0, {"status": "slow", "cos_values": []}

        # 궤적 포인트 샘플링 → 투표
        n_points = min(len(traj), 8)                   # 샘플링 포인트 수 (최대 8)
        step = max(1, len(traj) // n_points)           # 궤적 샘플링 간격

        agree = 0            # 흐름과 같은 방향 카운트
        disagree = 0         # 흐름과 반대 방향 카운트
        skip = 0             # flow 없음 등으로 스킵한 포인트 수
        debug_points = []    # 디버그용 포인트/내적/결과 저장
        cos_values = []      # 내적값 리스트 (디버그/표시용)

        for idx in range(0, len(traj), step):
            px, py = traj[idx]                                  # 궤적 포인트
            flow_v = self.flow.get_interpolated(px, py)         # 그 지점의 흐름 벡터

            if flow_v is None:                                  # 유효한 흐름 없음
                skip += 1
                debug_points.append((px, py, 0, "skip"))        # 스킵으로 기록
                continue

            cos_sim = ndx * flow_v[0] + ndy * flow_v[1]         # 내적(코사인 유사도)
            cos_values.append(cos_sim)                          # 리스트에 저장

            if cos_sim < cfg.cos_threshold:                     # 역방향이면
                disagree += 1
                debug_points.append((px, py, cos_sim, "disagree"))
            else:                                               # 흐름과 대체로 비슷
                agree += 1
                debug_points.append((px, py, cos_sim, "agree"))

        total_checked = agree + disagree                        # 실제로 평가한 포인트 수
        debug_info = {
            "agree": agree,
            "disagree": disagree,
            "skip": skip,
            "total": total_checked,
            "points": debug_points,
            "threshold": adaptive_threshold,
            "status": "voting",
            "cos_values": cos_values,
        }

        if total_checked < 3:  # 샘플이 너무 적으면 판정 보류
            return False, 0, debug_info

        disagree_ratio = disagree / total_checked               # 역방향 비율

        if disagree_ratio >= cfg.vote_threshold:                # 역방향 비율이 임계치 이상
            # 첫 의심 시점 기록
            if track_id not in st.first_suspect_frame:
                st.first_suspect_frame[track_id] = st.frame_num
                print(f"   ⚠️ ID:{track_id} 역주행 의심 시작 "
                      f"(frame={st.frame_num}, "
                      f"첫등장={st.first_seen_frame.get(track_id, '?')})")

            st.wrong_way_count[track_id] += 1                   # 의심 카운트 증가

            if st.wrong_way_count[track_id] >= cfg.wrong_count_threshold:
                st.wrong_way_ids.add(track_id)                  # 역주행 확정
                return True, disagree_ratio, debug_info
        else:
            # 역주행 비율이 낮으면 카운트 감소 (0보다 작아지지 않게)
            st.wrong_way_count[track_id] = max(
                0, st.wrong_way_count[track_id] - 2
            )

        return False, disagree_ratio, debug_info