# mini_project_0313
Traffic Monitoring and Anomaly Detection


Frontend====================================================================================

cmd에서

# 프론트엔드 폴더로 이동
cd frontend_js

npm install

npm run dev


Backend=====================================================================================

backend_flask 안의 migrations 폴더 지우기

cmd에서

# 백엔드 폴더로 이동
cd backend_flask

# 새로운 가상환경 만들기
conda create -n tads python=3.11 -y
conda activate tads

pip install -r requirements.txt

# DB 초기화
workbench나 vscode database에서
CREATE DATABASE tads;

cmd에서 

flask db init

flask db migrate -m "message"

flask db upgrade

python app.py로 실행

Simulation/.env==============================================================

N드라이브/이동훈/assets 폴더 다운 받아서
backend_flask/ 에 넣기

env.txt 다운 받아서 프로젝트 폴더 최상단에 넣고 이름 .env로 바꾸기
