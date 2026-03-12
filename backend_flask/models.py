from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

# -------------------------------------------------------------------------
# 1. 사용자 테이블 (관리자 계정 관리용)
# -------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))

    @property
    def password(self):
        raise AttributeError("비밀번호는 직접 읽을 수 없습니다.")
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


# -------------------------------------------------------------------------
# 2. 통합 감지 결과 테이블 (부모)
# -------------------------------------------------------------------------
class DetectionResult(db.Model):
    __tablename__ = 'detection_results'
    id = db.Column(db.Integer, primary_key=True)
    
    # 구분: 'fire', 'reverse', 'manual'
    event_type = db.Column(db.String(20))  
    address = db.Column(db.String(200), default="서울시 관제구역")
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    detected_at = db.Column(db.DateTime, default=datetime.now)
    
    # ✅ 추가: 시뮬레이션 데이터 여부 구분 (True: 시뮬레이션, False: 실시간/CCTV)
    is_simulation = db.Column(db.Boolean, default=False, nullable=False)

    # 공통 조치 정보
    is_resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime, nullable=True)
    feedback = db.Column(db.Boolean, nullable=True) # True: 정탐, False: 오탐
    resolved_by = db.Column(db.String(50), db.ForeignKey('users.name'), nullable=True)
    resolver = db.relationship('User', backref='resolved_cases')
    # 🔗 상세 테이블들과의 1:1 관계 설정
    fire_detail = db.relationship('FireResult', backref='base_result', uselist=False, cascade="all, delete-orphan")
    reverse_detail = db.relationship('ReverseResult', backref='base_result', uselist=False, cascade="all, delete-orphan")
    manual_detail = db.relationship('ManualResult', backref='base_result', uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.event_type,
            "address": self.address,
            "lat": self.latitude,
            "lng": self.longitude,
            "time": self.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            "is_simulation": self.is_simulation,  # ✅ 프론트 필터링용
            "is_resolved": self.is_resolved,
            "feedback": self.feedback,
            "resolved_at": self.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if self.resolved_at else None,
            "resolved_by_name": self.resolver.name if self.resolver else self.resolved_by
        }


# -------------------------------------------------------------------------
# 3~5. 상세 테이블들 (기존과 동일)
# -------------------------------------------------------------------------
class FireResult(db.Model):
    __tablename__ = 'fire_results'
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    fire_severity = db.Column(db.String(20)) 

class ReverseResult(db.Model):
    __tablename__ = 'reverse_results'
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    vehicle_info = db.Column(db.String(50)) 

class ManualResult(db.Model):
    __tablename__ = 'manual_results'
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    memo = db.Column(db.Text, nullable=True)