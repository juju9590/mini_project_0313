# 1. 최상단: gevent 패치 (SocketIO와 멀티스레딩 호환을 위해 필수)
from gevent import monkey
monkey.patch_all()              # 영상을 계속 분석하면서 동시에 웹 요청도 받아야함 (비동기로 한 번에 여러 일을 막힘없이 처리)

import os
import warnings
import atexit  # ✅ 추가: 프로세스 종료 감지
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_migrate import Migrate 

# 2. 분석 매니저 임포트 (종료 시 사용)
from detectors.manager import detector_manager 
# 3. DB 객체 임포트
from models import db 

# 환경 변수 로드
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.distributed.reduce_op` is deprecated")

app = Flask(__name__)
CORS(app)

# DB 설정 값 로드
DB_USER = os.getenv("DB_USER")
DB_PW = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy & DB 설정
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "fallback-secret-key")

# 4. DB 및 마이그레이션 초기화
db.init_app(app)
migrate = Migrate(app, db)

# 5. SocketIO 초기화 (gevent 모드)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
app.extensions['socketio'] = socketio

# 6. 블루프린트 임포트 및 등록
from routes.member import member_bp
from routes.result import result_bp
from routes.its import its_bp
from routes.streaming import streaming_bp
from routes.simulation import simulation_bp

app.register_blueprint(member_bp, url_prefix='/api/member')
app.register_blueprint(result_bp)
app.register_blueprint(its_bp, url_prefix='/api/its')
app.register_blueprint(streaming_bp)
app.register_blueprint(simulation_bp)

# ✅ 서버 종료 시 실행될 함수 등록 (분석 스레드 안전 종료)
def shutdown_detectors():
    print("🛑 [System] 서버 종료 감지: 모든 분석 스레드를 정지합니다...")
    detector_manager.stop_all()

atexit.register(shutdown_detectors)     # 서버 갑자기 종료 시 모든 분석기 안전하게 정지시키는 장치

# 실시간 조치 동기화
@socketio.on('resolve_emergency')
def handle_resolve(data):
    print(f"📡 조치 신호 전파: {data.get('alertId')}")
    emit('emergency_resolved', data, broadcast=True)

# 7. 메인 라우트 정의
@app.route('/')
def index():
    return "TADS Backend Server is Running"

# 8. 앱 실행부
if __name__ == '__main__':
    # 캡처 이미지 저장 폴더 생성
    save_path = os.path.join(app.root_path, "static", "captures")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    port = int(os.getenv("PORT", 5000))
    # debug=True 사용 시 리로더 때문에 스레드가 중복 실행될 수 있으므로 주의
    socketio.run(app, host='0.0.0.0', port=port, debug=True)