/* eslint-disable */
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import Swal from 'sweetalert2';

function Register() {
  const [formData, setFormData] = useState({                                // form에 입력한 값이 하나의 객체로 저장
    name: '', id: '', password: '', phone: '', email: ''
  });
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });          // setFormData는 이미 적어놓은 정보들은 계속 버리지 않고 유지하면서 새로운 입력값들을 받겠다
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    try {
      const host = window.location.hostname;
      const res = await axios.post(`http://${host}:5000/api/member/register`, formData);
      
      if (res.data.success) {
        Swal.fire({
          icon: 'success',
          title: '등록 완료',
          text: '관리자 등록이 승인되었습니다.',
          background: '#1e293b',
          color: '#fff',
          confirmButtonColor: '#3b82f6',
          confirmButtonText: '로그인하러 가기'
        }).then(() => {
          navigate('/login');
        });
      }
    } catch (err) {
      const msg = err.response?.data?.message || "회원가입 중 오류가 발생했습니다.";
      setErrorMsg(msg);
    }
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h2 style={titleStyle}>관리자 등록</h2>
        <p style={subTitleStyle}>TADS 시스템 이용을 위한 정보 입력</p>
        
        <form onSubmit={handleRegister} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <input name="name" placeholder="관리자 성함" onChange={handleChange} required style={inputStyle} />
          <input name="id" placeholder="희망 아이디" onChange={handleChange} required style={inputStyle} />
          <input name="password" type="password" placeholder="비밀번호" onChange={handleChange} required style={inputStyle} />
          <input name="phone" placeholder="비상 연락처 (010-0000-0000)" onChange={handleChange} required style={inputStyle} />
          <input name="email" type="email" placeholder="이메일 주소" onChange={handleChange} required style={inputStyle} />
          
          {errorMsg && (
            <div style={errorBoxStyle}>
              ⚠️ {errorMsg}
            </div>
          )}

          <button type="submit" style={btnStyle}>등록 요청</button>
        </form>

        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <Link to="/login" style={{ fontSize: '13px', color: '#94a3b8', textDecoration: 'none' }}>이미 계정이 있으신가요? <span style={{color:'#3b82f6', fontWeight:'bold'}}>로그인</span></Link>
        </div>
      </div>
    </div>
  );
}

// 스타일 가이드 (Login과 일치)
const containerStyle = { display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: '#020617' };
const cardStyle = { padding: '40px', background: '#0f172a', borderRadius: '16px', border: '1px solid #1e293b', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)', width: '100%', maxWidth: '420px' };
const titleStyle = { textAlign: 'center', marginBottom: '8px', color: '#fff', fontSize: '22px', fontWeight: '800' };
const subTitleStyle = { textAlign: 'center', marginBottom: '30px', color: '#64748b', fontSize: '14px' };
const inputStyle = { padding: '13px', borderRadius: '10px', border: '1px solid #334155', background: '#020617', color: '#fff', outline: 'none', fontSize: '14px' };
const btnStyle = { padding: '14px', background: '#2563eb', color: 'white', border: 'none', borderRadius: '10px', cursor: 'pointer', fontWeight: 'bold', marginTop: '10px', fontSize: '15px' };
const errorBoxStyle = { color: '#f87171', fontSize: '13px', textAlign: 'center', background: 'rgba(248, 113, 113, 0.1)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(248, 113, 113, 0.2)' };

export default Register;