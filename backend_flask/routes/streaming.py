import os
import cv2
import gevent
from datetime import datetime
from flask import Blueprint, Response, request, jsonify, current_app
from ultralytics import YOLO
from models import db, DetectionResult, FireResult, ReverseResult, ManualResult
import routes.shared as shared # ✅ 공통 변수 참조
from yolo_models.yolo_model import yolo_fire

streaming_bp = Blueprint('streaming', __name__)

# 모델 로드
# MODEL_PATH = 'best_SB.pt'
# MODEL_PATH = '/app/best_SB_openvino_model'
model = yolo_fire

@streaming_bp.route('/api/capture_now', methods=['POST'])
def capture_now():
    try:
        data = request.get_json()
        video_type = data.get('type', 'webcam')
        admin_name = data.get('adminName', '관리자')

        is_sim = (video_type != 'webcam')
        
        if video_type == 'sim':
            video_type = shared.current_broadcast_type if shared.current_broadcast_type else next(iter(shared.latest_frames), 'webcam')

        frame = shared.latest_frames.get(video_type)
        if frame is None:
            return jsonify({"status": "error", "message": "영상을 찾을 수 없습니다."}), 400

        new_alert = DetectionResult(
            event_type='manual',
            address="관리자 수동 캡처 구역",
            latitude=shared.sim_coords["lat"],
            longitude=shared.sim_coords["lng"],
            is_resolved=True,
            feedback=True,
            resolved_at=datetime.now(),
            is_simulation=is_sim,
            resolved_by=admin_name
        )
        db.session.add(new_alert)
        db.session.flush()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_{new_alert.id}_{ts}.jpg"
        filepath = os.path.join(shared.CAPTURE_DIR, filename)
        cv2.imwrite(filepath, frame)

        manual_detail = ManualResult(
            result_id=new_alert.id,
            image_path=f"/static/captures/{filename}",
            memo="메모 없음"
        )
        db.session.add(manual_detail)
        db.session.commit()

        return jsonify({
            "status": "success",
            "db_id": new_alert.id, 
            "image_url": manual_detail.image_path
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# 📝 메모를 업데이트하는 API (수동 캡처 후 호출됨)
@streaming_bp.route('/api/update_capture_memo', methods=['POST'])
def update_capture_memo():
    try:
        data = request.get_json()
        db_id = data.get('db_id')
        memo = data.get('memo', '').strip()
        
        print(f"📝 [메모 업데이트 시도] ID: {db_id}, 내용: {memo}")

        if not db_id:
            return jsonify({"status": "error", "message": "ID가 누락되었습니다."}), 400

        # ManualResult 테이블에서 부모 ID(result_id)로 조회
        detail = ManualResult.query.filter_by(result_id=db_id).first()
        
        if detail:
            detail.memo = memo
            db.session.commit()
            print(f"✅ [메모 업데이트 성공] ID: {db_id}")
            return jsonify({"status": "success"}), 200
        else:
            print(f"⚠️ [메모 업데이트 실패] ID {db_id}를 찾을 수 없습니다.")
            return jsonify({"status": "error", "message": "기록을 찾을 수 없습니다."}), 404
            
    except Exception as e:
        db.session.rollback()
        print(f"❌ [메모 업데이트 에러]: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def gen_frames(video_type, socketio, app):
    is_sim = (video_type != 'webcam')

    if video_type == "webcam":
        os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"
        # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000|rtmp_transport;tcp|fflags;nobuffer|flags;low_delay"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "probesize;32|analyzeduration;0|rtmp_transport;tcp|fflags;nobuffer|flags;low_delay"
        
        rtmp_url = "ffmpeg:rtmp://127.0.0.1:19350/live/stream"
        hls_url = "http://rtmp:8080/hls/stream.m3u8"
        
        # camera = cv2.VideoCapture(hls_url)
        # camera = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 줄임 (실시간에서 지연 줄이는 옵션)
    else:
        file_name = shared.current_video_file.get(video_type, f"{video_type}.mp4")
        video_path = os.path.join(os.getcwd(), "assets", file_name)
        camera = cv2.VideoCapture(video_path)
    
    frame_count = 0
    skip_frames = 3 # 4프레임 중 1번만 실행
    last_plot = None
    fail_count = 0

    try:
        while True:
            # 시뮬레이션 중단 체크
            if video_type != "webcam" and shared.current_broadcast_type != video_type:
                break

            success, frame = camera.read()
            if not success:
                fail_count += 1
                gevent.sleep(0.1)
                if fail_count > 30: break
                continue

            fail_count = 0
            frame_count += 1
            display_frame = frame
            shared.latest_frames[video_type] = frame.copy()

            # 1️⃣ [화재 감지 로직]
            if video_type in ["fire", "webcam"]:
                if frame_count % (skip_frames + 1) == 0:
                    # results = model.predict(frame, conf=0.4, verbose=False)
                    frame_small = cv2.resize(frame, (640,640))
                    results = model(frame_small, conf=0.4, verbose=False)
                    last_plot = results[0].plot()   # yolo 안돌리는 프레임은 이전 결과 사용

                    if len(results[0].boxes) > 0 and not shared.alert_sent_session[video_type]:
                        shared.alert_sent_session[video_type] = True
                        curr_lat = 37.5413 if video_type == "webcam" else shared.sim_coords["lat"]
                        curr_lng = 126.8381 if video_type == "webcam" else shared.sim_coords["lng"]

                        with app.app_context():
                            new_alert = DetectionResult(
                                event_type='fire',
                                latitude=curr_lat, longitude=curr_lng,
                                address="서울시 화재 관제구역", is_simulation=is_sim, is_resolved=False
                            )
                            db.session.add(new_alert)
                            db.session.flush()

                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"fire_{new_alert.id}_{ts}.jpg"
                            filepath = os.path.join(shared.CAPTURE_DIR, filename)
                            cv2.imwrite(filepath, last_plot)

                            fire_detail = FireResult(
                                result_id=new_alert.id,
                                image_path=f"/static/captures/{filename}",
                                fire_severity="심각"
                            )
                            db.session.add(fire_detail)
                            db.session.commit()
                            db_id = new_alert.id

                        socketio.emit('anomaly_detected', {
                            "alert_id": db_id,
                            "type": shared.ANOMALY_DATA[video_type]['type'], 
                            "lat": curr_lat, "lng": curr_lng,
                            "video_origin": video_type
                        })
                display_frame = last_plot if last_plot is not None else frame

            # 2️⃣ [역주행 감지 로직]
            elif video_type == "reverse":
                if not shared.alert_sent_session["reverse"]:
                    if camera.get(cv2.CAP_PROP_POS_FRAMES) >= 10:
                        shared.alert_sent_session["reverse"] = True
                        with app.app_context():
                            new_alert = DetectionResult(
                                event_type="reverse",
                                latitude=shared.sim_coords["lat"], longitude=shared.sim_coords["lng"],
                                address="역주행 발생 구역", is_simulation=is_sim, is_resolved=False
                            )
                            db.session.add(new_alert)
                            db.session.flush()

                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"reverse_{new_alert.id}_{ts}.jpg"
                            filepath = os.path.join(shared.CAPTURE_DIR, filename)
                            cv2.imwrite(filepath, frame)

                            reverse_detail = ReverseResult(
                                result_id=new_alert.id,
                                image_path=f"/static/captures/{filename}",
                                vehicle_info="감지된 차량"
                            )
                            db.session.add(reverse_detail)
                            db.session.commit()
                            db_id = new_alert.id

                        socketio.emit('anomaly_detected', {
                            "alert_id": db_id, "type": "역주행", 
                            "lat": shared.sim_coords["lat"], "lng": shared.sim_coords["lng"], 
                            "video_origin": "reverse"
                        })
                display_frame = cv2.resize(display_frame, (640, 360))
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            gevent.sleep(0.001)
    finally:
        camera.release()

@streaming_bp.route('/api/video_feed')
def video_feed():
    video_type = request.args.get('type', 'fire')
    socketio = current_app.extensions['socketio']
    app = current_app._get_current_object() 
    return Response(gen_frames(video_type, socketio, app),
                    mimetype='multipart/x-mixed-replace; boundary=frame')