# mini_project_0313
Traffic Monitoring and Anomaly Detection


# Frontend

cmd

cd frontend_js

npm install

npm run dev

# Simulation/.env

N드라이브/이동훈/ 위치에서

assets 폴더 다운 받아서 backend_flask/ 에 넣기

env.txt 다운 받아서 프로젝트 폴더 최상단에 넣고 이름 .env로 바꾸기

# Backend

backend_flask 안의 migrations 폴더 지우기

cmd

cd backend_flask

-새로운 가상환경 만들기
conda create -n tads python=3.11 -y
conda activate tads

pip install -r requirements.txt

# DB
workbench나 vscode database에서
CREATE DATABASE tads;

cmd

flask db init

flask db migrate -m "message"

flask db upgrade

python app.py로 실행
