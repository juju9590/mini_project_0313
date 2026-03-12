/* eslint-disable */
import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { io } from "socket.io-client";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Login from './pages/member/Login';
import Register from './pages/member/Register'; 
import Dashboard from './pages/traffic/Dashboard';

function App() {
  const [user, setUser] = useState(() => {
    const savedUser = sessionStorage.getItem('user'); 
    try {
      return savedUser && savedUser !== "undefined" ? JSON.parse(savedUser) : null;
    } catch { return null; }
  });

  const [socket, setSocket] = useState(null);

  const handleLogout = () => {
    sessionStorage.removeItem('user'); 
    setUser(null);
    if (socket) socket.disconnect();
  };

  useEffect(() => {
    if (user) {
      const host = window.location.hostname;
      const newSocket = io(`http://${host}:5000`, {
        transports: ["polling", "websocket"],
        forceNew: true,
        reconnectionAttempts: 3, 
        timeout: 5000,
      });

      newSocket.on("connect", () => {
        console.log(`✅ 서버(${host}:5000)와 소켓 연결 성공!`);
      });

      newSocket.on("disconnect", (reason) => {
        console.warn("⚠️ 서버와 연결이 끊어졌습니다:", reason);
        if (reason === "transport close" || reason === "io server disconnect") {
          toast.error("서버 세션이 만료되었습니다. 다시 로그인해주세요.");
          handleLogout();
        }
      });

      newSocket.on("connect_error", (error) => {
        console.error("❌ 서버 연결 실패 (서버가 꺼져 있음):", error);
        toast.error("서버에 연결할 수 없습니다. 다시 로그인해 주세요.");
        handleLogout(); 
      });

      setSocket(newSocket);
      return () => {
        if (newSocket) newSocket.close();
      };
    } else {
      setSocket(null);
    }
  }, [user]);

  return (
    <Router>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        body {
          margin: 0;
          padding: 0;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background-color: #020617;
          color: #ffffff;
          /* 🟢 수정: 모바일에서 전체 스크롤이 가능하도록 overflow를 유연하게 설정 */
          overflow-x: hidden;
          overflow-y: auto;
        }

        /* 🟢 여기에 추가: 토스트 너비 조절 및 줄바꿈 방지 */
        .Toastify__toast-container {
          width: 550px !important; /* 너비를 대폭 확장 */
        }
        .Toastify__toast-body {
          white-space: nowrap !important; /* 텍스트가 절대 아래로 꺽이지 않게 함 */
        }

        /* 데스크탑에서만 스크롤바를 숨기고 싶을 경우 사용 */
        @media (min-width: 1024px) {
          *::-webkit-scrollbar { display: none !important; }
          * { -ms-overflow-style: none !important; scrollbar-width: none !important; }
        }
      `}</style>

      <ToastContainer 
        position="top-right" 
        autoClose={2000} 
        theme="dark"
        pauseOnFocusLoss={false}
      />

      <div style={{ minHeight: '100dvh', background: '#020617' }}>
        <Routes>
          <Route 
            path="/" 
            element={user ? <Dashboard socket={socket} user={user} setUser={setUser} onLogout={handleLogout}/> : <Navigate to="/login" />} 
          />
          <Route 
            path="/login" 
            element={user ? <Navigate to="/" /> :                             // 메인('/')에 들어가기 전 유저정보가 있는 지 확인 없으면 로그인페이지로
              <Login setUser={(u) => {                                        // setUser는 변수가 아니라 나중에 로그인 성공 시 이 함수를 실행해서 정보를 저장해 라는 명령어 세트(함수)
                setUser(u); 
                sessionStorage.setItem('user', JSON.stringify(u)); 
              }} />
            } 
          />
          <Route path="/register" element={user ? <Navigate to="/" /> : <Register />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

