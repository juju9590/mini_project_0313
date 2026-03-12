from ultralytics import YOLO

# 1. 기존 .pt 모델 로드
model_fire = YOLO("../best_SB.pt")  # 화재 모델
model_reverse = YOLO("../best_DW.pt")  # 역주행 모델

# 2. OpenVINO 포맷으로 내보내기
# imgsz는 아까 논의한 대로 224나 320 정도로 고정하는 것이 좋습니다.
model_fire.export(format="openvino", imgsz=320)
model_reverse.export(format="openvino", imgsz=320)