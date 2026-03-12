from ultralytics import YOLO

# 역주행 모델
yolo_reverse = YOLO("yolo_models/best_DW_openvino_model", task='detect')

# 화재 모델
yolo_fire = YOLO("yolo_models/best_SB_openvino_model", task='detect')