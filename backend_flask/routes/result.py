from flask import Blueprint, jsonify, request
from models import db, DetectionResult, User
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import joinedload

result_bp = Blueprint('result', __name__)

@result_bp.route('/api/pending_alerts', methods=['GET'])
def get_pending_alerts():
    try:
        # DB에서 조치되지 않은 데이터 최신순 로드
        unresolved_results = DetectionResult.query.options(
            joinedload(DetectionResult.fire_detail),
            joinedload(DetectionResult.reverse_detail),
            joinedload(DetectionResult.manual_detail),
            joinedload(DetectionResult.resolver)
        ).filter_by(is_resolved=False).order_by(DetectionResult.detected_at.desc()).all()
        
        pending_list = []
        for res in unresolved_results:
            # ✅ 관계 설정(backref)을 활용하면 쿼리를 또 날릴 필요 없이 바로 접근 가능합니다.
            image_path = None
            if res.event_type == 'fire' and res.fire_detail:
                image_path = res.fire_detail.image_path
            elif res.event_type == 'reverse' and res.reverse_detail:
                image_path = res.reverse_detail.image_path
            # 수동 캡처(manual)는 보통 생성 즉시 resolved 처리하지만, 목록 조회가 필요할 경우를 대비
            elif res.event_type == 'manual' and res.manual_detail:
                image_path = res.manual_detail.image_path

            pending_list.append({
                "id": res.id,
                "type": "화재 발생" if res.event_type == 'fire' else "역주행" if res.event_type == 'reverse' else "수동 기록",
                "address": res.address,
                "time": res.detected_at.strftime('%p %I:%M:%S'),
                "lat": res.latitude,
                "lng": res.longitude,
                "origin": res.event_type,
                "image_url": image_path,
                "resolved_by": res.resolved_by                
            })
            
        return jsonify(pending_list), 200
    except Exception as e:
        print(f"❌ [미조치 목록 로드 에러]: {e}")
        return jsonify({"error": str(e)}), 500

@result_bp.route('/api/resolve_alert_db', methods=['POST'])
def resolve_alert_db():
    try:
        data = request.json
        alert_id = data.get('alertId')
        is_correct = data.get('isCorrect') 
        admin_name = data.get('adminName', '').strip() if data.get('adminName') else 'Unknown'
        is_simulation = data.get('is_simulation', 0)
        user_exists = User.query.filter_by(name=admin_name).first()

        result = DetectionResult.query.get(alert_id)
        if result:
            result.is_resolved = True
            result.resolved_at = datetime.now()
            result.feedback = True if is_correct == 1 else False
            if user_exists:
                result.resolved_by = admin_name
            else:
                print(f"⚠️ [주의] 유저 '{admin_name}'가 DB에 없어 resolved_by를 비워둡니다.")
                result.resolved_by = None
            result.is_simulation = is_simulation
            
            db.session.commit()
            status_msg = "정탐" if result.feedback else "오탐"
            print(f"✅ [DB 업데이트 성공] ID {alert_id} 조치 완료 ({status_msg})")
            return jsonify({"success": True, "feedback": status_msg}), 200
        else:
            return jsonify({"success": False, "message": "알림을 찾을 수 없습니다."}), 404
    except Exception as e:
        db.session.rollback()
        print(f"❌ [DB 업데이트 에러]: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
@result_bp.route('/api/resolve_alerts_bulk', methods=['POST'])
def resolve_alerts_bulk():
    try:
        data = request.json
        alert_ids = data.get('alertIds', [])  # ID 리스트를 받음
        is_correct = data.get('isCorrect', 1) 
        admin_name = data.get('adminName', 'Unknown')
        is_simulation = data.get('is_simulation', 0)

        if not alert_ids:
            return jsonify({"success": False, "message": "조치할 ID가 없습니다."}), 400

        # 1. 해당 ID들에 해당하는 모든 데이터를 한 번에 가져옴
        results = DetectionResult.query.filter(DetectionResult.id.in_(alert_ids)).all()
        
        now = datetime.now()
        for result in results:
            result.is_resolved = True
            result.resolved_at = now
            result.feedback = True if is_correct == 1 else False
            result.resolved_by = admin_name
            result.is_simulation = is_simulation
        
        # 2. 단 한 번의 커밋으로 모든 수정사항 반영 (매우 안전)
        db.session.commit()
        
        print(f"✅ [일괄 업데이트 성공] {len(results)}건 조치 완료 (by {admin_name})")
        return jsonify({"success": True, "count": len(results)}), 200

    except Exception as e:
        db.session.rollback()
        print(f"❌ [일괄 업데이트 에러]: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@result_bp.route('/api/stats/summary', methods=['GET'])
def get_stats_summary():
    try:
        # 1. 쿼리 파라미터 확인 (mode: 'real', 'sim', 'all' 중 선택 / 기본값 'real')
        mode = request.args.get('mode', 'real')

        # 2. 모드에 따른 기본 쿼리 필터링 설정
        if mode == 'sim':
            base_query = DetectionResult.query.filter_by(is_simulation=True)
        elif mode == 'all':
            base_query = DetectionResult.query  # 필터 없음
        else: # 'real' (기본값)
            base_query = DetectionResult.query.filter_by(is_simulation=False)

        # 3. 데이터 집계 (선택된 모드 기준)
        total_resolved = base_query.filter_by(is_resolved=True).count()
        correct_count = base_query.filter_by(is_resolved=True, feedback=True).count()
        incorrect_count = base_query.filter_by(is_resolved=True, feedback=False).count()
        
        fire_count = base_query.filter_by(event_type='fire').count()
        reverse_count = base_query.filter_by(event_type='reverse').count()
        manual_count = base_query.filter_by(event_type='manual').count()
        
        # 4. 정확도 계산
        precision = 0
        if total_resolved > 0:
            precision = round((correct_count / total_resolved) * 100, 1)
            
        return jsonify({
            "current_mode": mode,
            "total": total_resolved,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "precision": precision,
            "type_counts": {
                "fire": fire_count,
                "reverse": reverse_count,
                "manual": manual_count
            }
        }), 200
        
    except Exception as e:
        print(f"❌ [통계 데이터 로드 에러]: {e}")
        return jsonify({"error": str(e)}), 500
    

@result_bp.route('/api/stats/history', methods=['GET'])
def get_stats_history():
    try:
        # 1. 날짜 및 모드 파라미터 받기
        target_date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        mode = request.args.get('mode', 'all') # 이력은 보통 '전체'를 기본으로 보는 경우가 많음

        # 2. 날짜 필터 적용
        query = DetectionResult.query.options(
            joinedload(DetectionResult.fire_detail),
            joinedload(DetectionResult.reverse_detail),
            joinedload(DetectionResult.manual_detail),
            joinedload(DetectionResult.resolver)
        ).filter(func.date(DetectionResult.detected_at) == target_date_str)

        # 3. 모드 필터 적용
        if mode == 'real':
            query = query.filter_by(is_simulation=False)
        elif mode == 'sim':
            query = query.filter_by(is_simulation=True)

        results = query.order_by(DetectionResult.detected_at.desc()).all()
        
        history_list = []
        for res in results:
            image_path = None
            memo = None
            
            if res.event_type == 'fire' and res.fire_detail:
                image_path = res.fire_detail.image_path
            elif res.event_type == 'reverse' and res.reverse_detail:
                image_path = res.reverse_detail.image_path
            elif res.event_type == 'manual' and res.manual_detail:
                image_path = res.manual_detail.image_path
                memo = res.manual_detail.memo

            history_list.append({
                "id": res.id,
                "type": res.event_type,
                "is_simulation": res.is_simulation,
                "address": res.address,
                "time": res.detected_at.strftime('%H:%M:%S'),
                "image_path": image_path,
                "memo": memo,
                "feedback": res.feedback,
                "resolved_by": res.resolved_by
            })
            
        return jsonify({"logs": history_list}), 200
        
    except Exception as e:
        print(f"❌ [이력 데이터 로드 에러]: {e}")
        return jsonify({"error": str(e)}), 500
    
@result_bp.route('/api/update_address', methods=['POST'])
def update_address():
    try:
        data = request.json
        alert_id = data.get('alertId')
        real_address = data.get('address')
        
        # DB에서 해당 ID의 데이터를 찾습니다.
        result = DetectionResult.query.get(alert_id)
        
        if result:
            # "서울시 화재 관제구역" 등으로 저장된 임시 주소를 실제 주소로 업데이트
            result.address = real_address
            db.session.commit()
            print(f"📍 [주소 업데이트 성공] ID: {alert_id} -> {real_address}")
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "message": "해당 알림 ID를 찾을 수 없습니다."}), 404
            
    except Exception as e:
        print(f"❌ [주소 업데이트 에러]: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
@result_bp.route('/api/resolve_alert', methods=['POST'])
def resolve_alert():
    # 1. global 키워드 대신 shared 모듈을 직접 가져옵니다.
    import routes.shared as shared 
    
    data = request.get_json()
    video_type = data.get('type')
    
    # 2. shared.변수명 형태로 접근하여 수정합니다.
    if video_type in shared.alert_sent_session and shared.alert_sent_session[video_type] == True:
        shared.alert_sent_session[video_type] = False
        
        if video_type != "webcam":
            shared.current_broadcast_type = None 
            
        return jsonify({"status": "success", "message": "Active session resolved"}), 200
    else:
        return jsonify({"status": "success", "message": "History record resolved"}), 200