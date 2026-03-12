/* eslint-disable */
import React, { useState } from 'react';

function Sidebar({ activeTab, onTabChange, user, onLogout, isMobile }) {
  // ✨ 1. 기본 상태를 접힘(true)으로 변경
  const [isCollapsed, setIsCollapsed] = useState(true);

  const menuItems = [
    // { id: "sim", label: "현장 시뮬레이션", icon: "🛡️" }, 
    { id: "cctv", label: "CCTV 모니터링", icon: "📡" },
    { id: "webcam", label: "웹캠 관제", icon: "📷" },
    { id: "stats", label: "통계 데이터", icon: "📊" }
  ];

  const sidebarWidth = isMobile ? '100%' : (isCollapsed ? '80px' : '240px');

  return (
    <aside style={{
      ...sidebarStyle,
      width: sidebarWidth,
      minWidth: isMobile ? 'auto' : sidebarWidth,
      height: isMobile ? 'auto' : '100vh',
      transition: 'width 0.35s cubic-bezier(0.4, 0, 0.2, 1)', // 애니메이션 곡선 최적화
      position: 'relative'
    }}>
      
      {/* ✨ 2. 더 이쁜 SVG 토글 버튼 */}
      {!isMobile && (
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          style={toggleButtonStyle}
        >
          <svg 
            width="14" 
            height="14" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2.5" 
            strokeLinecap="round" 
            strokeLinejoin="round"
            style={{ 
              transform: isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)', 
              transition: 'transform 0.3s' 
            }}
          >
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
        </button>
      )}

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        {/* 로고 영역 */}
        <div style={{ ...logoArea, padding: isMobile ? '12px 15px' : '30px 20px', justifyContent: isCollapsed ? 'center' : 'flex-start' }}>
          <div style={{...logoIcon, width: isMobile ? '32px' : '40px', height: isMobile ? '32px' : '40px', minWidth: isMobile ? '32px' : '40px'}}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{width: isMobile ? '18px' : '24px', color: '#6366f1'}}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          </div>
          {(!isCollapsed || isMobile) && (
            <div style={{ marginLeft: '12px', animation: 'fadeIn 0.3s' }}>
              <div style={{...logoTitle, fontSize: isMobile ? '18px' : '22px'}}>TADS</div>
              {!isMobile && <div style={{...logoSub, fontSize: '11px'}}>관제 센터 시스템</div>}
            </div>
          )}
        </div>
        
        {/* 메뉴 리스트 */}
        <nav style={{ 
          ...sideNavStyle, 
          flexDirection: isMobile ? 'row' : 'column', 
          padding: isMobile ? '0 10px 10px 10px' : '10px 0'
        }}>
          {!isMobile && !isCollapsed && <div style={{...menuLabel, fontSize: '12px', marginBottom: '15px'}}>메인 메뉴</div>}
          
          {menuItems.map((menu) => (
            <div key={menu.id} onClick={() => onTabChange(menu.id)} style={{
              ...menuItemStyle,
              padding: isMobile ? '5px' : '18px 0',
              justifyContent: (isCollapsed && !isMobile) ? 'center' : 'flex-start',
              paddingLeft: (isCollapsed || isMobile) ? '0' : '25px',
              backgroundColor: activeTab === menu.id ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
              color: activeTab === menu.id ? '#818cf8' : '#94a3b8',
              borderLeft: (!isMobile && !isCollapsed && activeTab === menu.id) ? '4px solid #6366f1' : 'none',
              borderRadius: isMobile ? '8px' : '0',
            }} title={isCollapsed ? menu.label : ""}>
              <span style={{ 
                marginRight: (isCollapsed || isMobile) ? '0px' : '12px', 
                fontSize: isMobile ? '18px' : '20px',
                width: isCollapsed ? '100%' : 'auto',
                textAlign: 'center'
              }}>{menu.icon}</span>
              {(!isCollapsed || isMobile) && (
                <span style={{ fontWeight: activeTab === menu.id ? '700' : '500', fontSize: isMobile ? '10px' : '15px', whiteSpace: 'nowrap' }}>
                  {menu.label}
                </span>
              )}
            </div>
          ))}
        </nav>
      </div>

      {!isMobile && (
        <div style={{
          ...sidebarFooter, 
          textAlign: isCollapsed ? 'center' : 'left', 
          padding: isCollapsed ? '20px 10px' : '20px 25px'
        }}>
          {isCollapsed ? (
            /* 접혔을 때는 상태 점(Dot)만 중앙에 표시 */
            <div style={{...statusDot, margin: '0 auto 15px auto'}} title={`${user?.name} 관리자님 접속 중`} />
          ) : (
            /* 펼쳐졌을 때는 관리자 이름과 상태 표시를 세로로 나열 */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '15px' }}>
              <div style={{ 
                fontSize: '12px', 
                color: '#818cf8', 
                fontWeight: '700',
                display: 'flex',
                alignItems: 'center',
                gap: '5px'
              }}>
                <span style={{ fontSize: '14px' }}>👤</span> {user?.name} 관리자
              </div>
              <div style={statusWrapper}>
                <div style={statusDot} />
                <span style={statusText}>시스템 온라인</span>
              </div>
            </div>
          )}

          {/* 로그아웃 버튼 */}
          <button onClick={onLogout} style={{
            ...logoutBtn, 
            fontSize: isCollapsed ? '10px' : '13px', 
            padding: isCollapsed ? '8px 0' : '10px',
            marginTop: isCollapsed ? '0' : '5px'
          }}>
            {isCollapsed ? 'Exit' : '로그아웃'}
          </button>
        </div>
      )}
    </aside>
  );
}

// 스타일 설정
const sidebarStyle = { background: '#0f172a', borderRight: '1px solid #1e293b', display: 'flex', flexDirection: 'column', zIndex: 100 };
const logoArea = { display: 'flex', alignItems: 'center', overflow: 'hidden' };
const logoIcon = { background: '#1e293b', borderRadius: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' };
const logoTitle = { fontWeight: '900', color: '#fff', letterSpacing: '1px' };
const logoSub = { color: '#64748b', whiteSpace: 'nowrap' };
const sideNavStyle = { display: 'flex' };
const menuLabel = { color: '#475569', padding: '0 25px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' };
const menuItemStyle = { cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'all 0.2s ease' };
const sidebarFooter = { borderTop: '1px solid #1e293b' };
const statusWrapper = { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' };
const statusDot = { width: '8px', height: '8px', background: '#22c55e', borderRadius: '50%', boxShadow: '0 0 8px #22c55e' };
const statusText = { fontSize: '11px', color: '#94a3b8', fontWeight: '500' };
const logoutBtn = { width: '100%', background: '#1e293b', border: '1px solid #334155', color: '#94a3b8', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s' };

const toggleButtonStyle = {
  position: 'absolute',
  right: '-12px',
  top: '40px',
  width: '26px',
  height: '26px',
  borderRadius: '50%',
  background: '#6366f1',
  color: 'white',
  border: '2px solid #0f172a',
  cursor: 'pointer',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  zIndex: 101,
  boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
  padding: 0
};

export default Sidebar;