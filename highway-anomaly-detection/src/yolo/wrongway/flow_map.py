# 15x15 그리드 기반 정상 흐름장
# EMA 학습 + 이중 선형 보간(Bilinear Interpolation) + 공간 평활화

import numpy as np
from pathlib import Path


class FlowMap:
    def __init__(self, grid_size: int, alpha: float, min_samples: int):
        self.grid_size = grid_size      # 흐름 맵을 나눌 격자 크기 (N x N)
        self.alpha = alpha              # EMA 학습 속도 (새 데이터 반영 비율)
        self.min_samples = min_samples  # 셀당 최소 학습 샘플 수 (이하이면 공간 보정)

        # 각 셀의 정상 이동 방향 벡터 (ndx, ndy)
        self.flow = np.zeros((grid_size, grid_size, 2), np.float32)
        # 각 셀의 학습 데이터 개수
        self.count = np.zeros((grid_size, grid_size), np.int32)

        self.frame_w = 0    # 영상 너비
        self.frame_h = 0    # 영상 높이
        self.cell_w = 1.0   # 셀 하나의 너비
        self.cell_h = 1.0   # 셀 하나의 높이

    # ==================== 초기화/리셋 ====================
    def init_grid(self, frame_w, frame_h):
        """영상 해상도에 맞게 flow_map 그리드 셀 크기 설정"""
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cell_w = frame_w / self.grid_size
        self.cell_h = frame_h / self.grid_size

    def reset(self):
        """flow_map과 count를 0으로 초기화"""
        self.flow[:] = 0
        self.count[:] = 0

    # ==================== 좌표 변환 ====================
    def _cell_coords(self, x, y):
        """픽셀 좌표(x, y)를 그리드 좌표(r, c)로 변환"""
        r = (y / self.cell_h) - 0.5  # 행 인덱스 실수 값 (0~grid_size-1 부근)
        c = (x / self.cell_w) - 0.5  # 열 인덱스 실수 값
        return r, c

    # ==================== EMA 기반 학습 ====================
    def learn_step(self, x1, y1, x2, y2, min_move):
        """한 차량의 이동 벡터를 flow_map에 반영 (EMA 기반 학습)"""

        dx, dy = x2 - x1, y2 - y1          # 이동 벡터
        mag = np.sqrt(dx ** 2 + dy ** 2)    # 크기(속도)
        if mag < min_move:
            return                          # 너무 작은 움직임은 무시

        ndx, ndy = dx / mag, dy / mag       # 단위 방향 벡터
        r = int((y1 + y2) / 2 / self.cell_h)  # 중간 위치의 셀 행 좌표
        c = int((x1 + x2) / 2 / self.cell_w)  # 중간 위치의 셀 열 좌표
        r = np.clip(r, 0, self.grid_size - 1)  # 범위 보정
        c = np.clip(c, 0, self.grid_size - 1)

        # EMA: 기존 흐름 벡터에 새 방향을 비율(alpha)만큼 섞어줌
        self.flow[r, c, 0] = (1 - self.alpha) * self.flow[r, c, 0] + self.alpha * ndx
        self.flow[r, c, 1] = (1 - self.alpha) * self.flow[r, c, 1] + self.alpha * ndy
        self.count[r, c] += 1  # 샘플 수 증가

    # ==================== 이중 선형 보간 ====================
    def get_interpolated(self, x, y):
        """이중 선형 보간으로 (x, y) 위치의 흐름 벡터 추정"""

        r, c = self._cell_coords(x, y)              # 그리드 좌표로 변환 (실수)
        r0, c0 = int(np.floor(r)), int(np.floor(c))  # 좌상단 셀 인덱스 (정수)
        r1, c1 = r0 + 1, c0 + 1                      # 우하단 셀 인덱스
        dr, dc = r - r0, c - c0                       # 소수점 부분(보간 가중치)

        # grid 범위 안으로 인덱스 보정
        gs = self.grid_size - 1
        r0, r1 = np.clip([r0, r1], 0, gs)
        c0, c1 = np.clip([c0, c1], 0, gs)

        v00 = self.flow[r0, c0]  # (r0, c0) 셀의 흐름 벡터
        v01 = self.flow[r0, c1]  # (r0, c1)
        v10 = self.flow[r1, c0]  # (r1, c0)
        v11 = self.flow[r1, c1]  # (r1, c1)

        top = (1 - dc) * v00 + dc * v01        # 위쪽 두 셀 가로 보간
        bottom = (1 - dc) * v10 + dc * v11     # 아래쪽 두 셀 가로 보간
        final_v = (1 - dr) * top + dr * bottom  # 위/아래 세로 보간

        mag = np.linalg.norm(final_v)           # 최종 벡터 크기
        # 크기가 충분하면 단위 벡터, 아니면 None
        return final_v / (mag + 1e-6) if mag > 0.1 else None

    # ==================== 공간 평활화 ====================
    def apply_spatial_smoothing(self):
        """데이터가 부족한 셀은 주변 3x3 이웃의 평균값으로 보정"""

        new_map = self.flow.copy()  # 새로운 맵 복사본
        for r in range(self.grid_size):          # 모든 칸을 순회
            for c in range(self.grid_size):
                if self.count[r, c] < self.min_samples:  # 샘플 부족한 셀만 처리
                    r_s = max(0, r - 1)                  # 주변 행 범위 시작
                    r_e = min(self.grid_size, r + 2)     # 주변 행 범위 끝
                    c_s = max(0, c - 1)                  # 주변 열 범위 시작
                    c_e = min(self.grid_size, c + 2)     # 주변 열 범위 끝
                    avg_v = np.mean(self.flow[r_s:r_e, c_s:c_e], axis=(0, 1))  # 주변 평균 벡터
                    if np.linalg.norm(avg_v) > 0.1:  # 유효한 벡터면
                        new_map[r, c] = avg_v        # 셀을 주변 평균으로 채움

        self.flow = new_map  # 맵 갱신

    # ==================== 저장/로드 ====================
    def save(self, path: Path):
        """학습된 flow_map과 count 정보를 .npy 파일로 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, {"flow": self.flow, "count": self.count})
        print(f"✅ flow_map 저장: {path}")

    def load(self, path: Path) -> bool:
        """기존에 저장된 flow_map 파일이 있으면 불러와서 사용"""
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=True).item()
        loaded = data["flow"]
        if loaded.shape[0] != self.grid_size:  # grid_size가 바뀐 경우 무시
            print("⚠️ grid 불일치, 초기화")
            return False
        self.flow = loaded             # flow_map 로드
        self.count = data["count"]     # count 로드
        print(f"✅ flow_map 로드 ({self.count.sum()} 샘플)")
        return True