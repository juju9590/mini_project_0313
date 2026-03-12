/* eslint-disable */
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import Swal from 'sweetalert2';

function Login({ setUser }) {                                                  // 처음에 setUser에는 아무 거도 없고 로그인 성공 시 여기에 상태를 저장한다
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    try {
      const host = window.location.hostname;
      const res = await axios.post(`http://${host}:5000/api/member/login`, {    // 서버에 데이터를 담아 보낸다 (이 정보로 로그인이 가능한가?)
        id: userId,                                                             // await : 서버의 대답이 올 떄까지 기다리고 다음 코드 실행해
        password: password 
      });
      
      if (res.data.success) {
        Swal.fire({
          title: '시스템 접속',
          text: `${res.data.user.name} 관리자님, 환영합니다.`,
          icon: 'success',
          background: '#1e293b',
          color: '#fff',
          confirmButtonColor: '#3b82f6',
          confirmButtonText: '대시보드 진입',
          heightAuto: false
        }).then((result) => {
          if (result.isConfirmed) {
            setUser(res.data.user);
            navigate('/');          // 로그인이 성공하면 메인화면으로 보내기 '/'
          }
        });
      }
    } catch (err) {
      const msg = err.response?.data?.message || "서버와 연결할 수 없습니다.";
      setErrorMsg(msg); 
    }
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <div style={logoWrapper}>
          <div style={logoIcon}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{width: '30px', color: '#6366f1'}}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          </div>
          <h2 style={logoText}>TADS</h2>
        </div>
        <p style={subTitleStyle}>지능형 교통 관제 시스템</p>
        
        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={inputGroupStyle}>
            <label style={labelStyle}>관리자 아이디</label>
            <input 
              type="text" 
              placeholder="Username" 
              value={userId}
              onChange={(e) => setUserId(e.target.value)}   // 사용자가 키보드를 칠 때마다 그 글자를 userid라는 변수에 실시간으로 담기
              style={inputStyle}
              required
            />
          </div>
          
          <div style={inputGroupStyle}>
            <label style={labelStyle}>비밀번호</label>
            <input 
              type="password" 
              placeholder="Password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={inputStyle}
              required
            />
          </div>

          {errorMsg && (
            <div style={errorBoxStyle}>
              ⚠️ {errorMsg}
            </div>
          )}
          
          <button type="submit" style={btnStyle}>시스템 로그인</button>
        </form>

        <div style={{ marginTop: '25px', textAlign: 'center' }}>
          <span style={{ fontSize: '13px', color: '#64748b' }}>신규 관리자이신가요? </span>
          <Link to="/register" style={linkStyle}>계정 등록</Link>
        </div>
      </div>
    </div>
  );
}

// 스타일 정의 (Dashboard 컨셉과 통일)
const containerStyle = { display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: '#020617' };
const cardStyle = { padding: '50px', background: '#0f172a', borderRadius: '16px', border: '1px solid #1e293b', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)', width: '100%', maxWidth: '400px' };
const logoWrapper = { display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '8px' };
const logoIcon = { width: '50px', height: '50px', background: '#1e293b', borderRadius: '10px', display: 'flex', justifyContent: 'center', alignItems: 'center', border: '1px solid #334155' };
const logoText = { fontSize: '40px', fontWeight: '800', color: '#fff', letterSpacing: '1px' };
const subTitleStyle = { textAlign: 'center', marginBottom: '35px', color: '#64748b', fontSize: '15px', fontWeight: '500' };
const inputGroupStyle = { display: 'flex', flexDirection: 'column', gap: '8px' };
const labelStyle = { fontSize: '13px', fontWeight: '600', color: '#94a3b8', marginLeft: '4px' };
const inputStyle = { padding: '14px', borderRadius: '10px', border: '1px solid #334155', background: '#020617', color: '#fff', outline: 'none', fontSize: '15px' };
const btnStyle = { padding: '15px', background: '#2563eb', color: 'white', border: 'none', borderRadius: '10px', cursor: 'pointer', fontWeight: 'bold', marginTop: '10px', fontSize: '15px', transition: 'background 0.2s' };
const linkStyle = { fontSize: '13px', color: '#3b82f6', textDecoration: 'none', fontWeight: 'bold' };
const errorBoxStyle = { color: '#f87171', fontSize: '13px', textAlign: 'center', background: 'rgba(248, 113, 113, 0.1)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(248, 113, 113, 0.2)' };

export default Login;