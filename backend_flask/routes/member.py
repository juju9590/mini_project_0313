from flask import Blueprint, request, jsonify
from models import db, User

member_bp = Blueprint('member', __name__)

@member_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    user_id = data.get('id')
    password = data.get('password')
    phone = data.get('phone')
    email = data.get('email')

    try:
        # 아이디 중복 확인 (ORM 방식)
        existing_user = User.query.filter_by(user_id=user_id).first()
        if existing_user:
            return jsonify({"success": False, "message": "이미 존재하는 아이디입니다."}), 400

        # 사용자 등록
        new_user = User(
            name=name,
            user_id=user_id,
            password=password, # 내부적으로 해싱됨
            phone=phone,
            email=email
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"success": True, "message": "회원가입 성공!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@member_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get('id')
    password = data.get('password')

    try:
        # 로그인 확인 (ORM 방식)
        user = User.query.filter_by(user_id=user_id).first()

        if user and user.verify_password(password):
            return jsonify({"success": True, "user": {"name": user.name, "user_id": user.user_id}})
        else:
            return jsonify({"success": False, "message": "아이디 또는 비밀번호가 틀렸습니다."}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500