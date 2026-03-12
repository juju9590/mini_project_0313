/* eslint-disable */
import { useState, useEffect } from 'react';
import axios from 'axios';
import { useAnomalyDetection } from '../hooks/useAnomalyDetection';
import VideoPanel from './VideoPanel';
import MapPanel from './MapPanel';
import ControlPanel from './ControlPanel';
import Sidebar from './Sidebar';
import StatsPanel from './StatsPanel';

function Dashboard({ socket, user, setUser, onLogout }) {             // socket : 실시간 사고 알림을 받는 무전기 역할, user : 현재 누가 로그인하였는 지
  const [activeTab, setActiveTab] = useState("cctv");                 // 현재 활성화된 탭 (sim, webcam, cctv, stats)
  const [videoUrl, setVideoUrl] = useState("");                       // 비디오 스트리밍 주소
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1024); // 모바일 여부 체크
  const [cctvData, setCctvData] = useState([]);                       // CCTV 목록 데이터
  const [showMap, setShowMap] = useState(true);

  const host = window.location.hostname;

  // 백엔드 API로부터 공공 CCTV 목록과 스트리밍 URL 정보 가져옴
  const fetchCctvUrl = async () => {
      try {
          const response = await axios.get(`http://${host}:5000/api/its/get_cctv_url`);
          if (response.data.success) {
              // 이제 cctvUrl 하나가 아니라 cctvData 배열을 받습니다.
              setCctvData(response.data.cctvData);
          }
      } catch (err) {
          console.error("CCTV API 호출 실패:", err);
      }
  };

  // 소켓을 통한 실시간 이상 탐지 데이터 관리
  const { isEmergency, pendingAlerts, logs, mapRef, resolveEmergency, resolveAllAlertsAction, moveToAlert } = 
    useAnomalyDetection(socket, activeTab, setActiveTab, setVideoUrl, host, user?.name);

  // 창 크기에따라 모바일/데스크탑 모드 전환
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 탭 전환시 지도 깨짐 방지, 사고 발생 시 최신 사고 지점으로 지도 이동
  useEffect(() => {
    if (activeTab !== "stats" && mapRef.current) {
      // 1. 지도가 display: block/flex로 바뀌는 순간 즉시 보정
      mapRef.current.relayout();

      if (isEmergency && pendingAlerts.length > 0) {
        setShowMap(true); 
      }

      const timer = setTimeout(() => {
        mapRef.current.relayout();
        
        // 2. 사고 발생 시 혹은 탭 전환 시 화면 이동 로직
        if (isEmergency && pendingAlerts.length > 0) {
          // ✨ 핵심: [length - 1](가장 오래된 것) 대신 [0](가장 최신 것)을 바라보게 합니다.
          // 이렇게 하면 새로운 사고가 터졌을 때 '돌아가는' 현상이 사라집니다.
          const latestAlert = pendingAlerts[0]; 
          const moveLatLng = new window.kakao.maps.LatLng(latestAlert.lat, latestAlert.lng);
          
          // 지도가 이미 그 근처라면 굳이 또 이동하지 않도록 로직을 보호해도 좋습니다.
          mapRef.current.panTo(moveLatLng); 
        }
      }, 300); 
      
      return () => clearTimeout(timer);
    }
    // isEmergency를 다시 넣어서 지도가 나타나는 시점을 잡되, 
    // 내부 로직이 '최신 알림'을 보게 함으로써 튕기는 버그만 잡는 것입니다.
  }, [activeTab, isEmergency, pendingAlerts.length]);

  if (!user) return null; 

  // 세션 정보 삭제하고 로그인 페이지로 리다이렉트
  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    } else {
      sessionStorage.removeItem('user');
      setUser(null);
      window.location.href = "/login";
    }
  };

  useEffect(() => {
    // 최초 진입 시 activeTab이 cctv라면 데이터를 즉시 가져옴
    if (activeTab === "cctv") {
      fetchCctvUrl();
    }
  }, []);
  
  // 사이드바 탭 클릭 시 상태 변경 및 데이터 로드
  const handleTabChange = (tab) => {
    setVideoUrl(""); 
    setActiveTab(tab);

    if (tab === "webcam" || tab === "sim") { 
      setTimeout(() => {
        // 탭 이름을 type으로 넘겨서 백엔드 gen_frames가 돌게
        setVideoUrl(`http://${host}:5000/api/video_feed?type=${tab}&v=${Date.now()}`);
      }, 100);
    }
    else if (tab === "cctv") {
      fetchCctvUrl();
    }
  };

  // 백엔드에 시뮬레이션 데이터 생성 요청
  const startSim = (type) => {
    axios.post(`http://${host}:5000/api/start_simulation`, { type })
      .catch(err => console.error("시뮬레이션 시작 실패:", err));
  };

  // 모든 알림 일괄 처리
  const resolveAllAlerts = () => {
    if (pendingAlerts.length === 0) return;
    // 루프를 돌리지 않고, 전체 배열(pendingAlerts)을 한 번 넘깁니다.
    // 훅 내부에서 확인창을 한 번 띄우고 알아서 루프를 돕니다.
    const isSimMode = activeTab === "sim";
    resolveAllAlertsAction(pendingAlerts, isSimMode);
  };

  // 알림 리스트에서 특정 알림 클릭 시 지도 위치 이동
  const handleAlertClick = (alert) => {
    moveToAlert(alert);
    if (mapRef.current) {
      mapRef.current.setLevel(2, { anchor: new window.kakao.maps.LatLng(alert.lat, alert.lng), animate: true });
    }
  };

  return (
    <div style={{ 
      ...containerStyle, 
      flexDirection: isMobile ? 'column' : 'row',
      height: isMobile ? 'auto' : '100vh', 
      overflow: isMobile ? 'visible' : 'hidden', 
      animation: isEmergency ? 'emergency-bg 0.8s infinite' : 'none' 
    }}>
      <style>{pulseAnimation + (isMobile ? "" : hideScrollbar)}</style>
      
      <Sidebar 
        activeTab={activeTab} 
        onTabChange={handleTabChange} 
        user={user} 
        onLogout={handleLogout} 
        isMobile={isMobile}
      />

      <main style={{ 
        ...mainWrapper, 
        height: isMobile ? 'auto' : '100vh', 
        padding: isMobile ? '10px' : '15px', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: isMobile ? 'visible' : 'hidden' 
      }}>
        {/* <header style={{...contentHeader, flexShrink: 0, marginBottom: isMobile ? '10px' : '10px'}}>
          <h2 style={{...headerTitle, fontSize: isMobile ? '18px' : '24px'}}>
            {activeTab === "sim" ? "현장 시뮬레이션" : 
            activeTab === "webcam" ? "실시간 웹캠 관제" :
            activeTab === "cctv" ? "공공 CCTV 모니터링" : "통계 데이터 분석"}
          </h2>
          {!isMobile && <div style={adminInfo}>{user?.name} 관리자님 접속 중</div>}
        </header> */}

        <div style={{ ...gridContainer, flexDirection: isMobile ? 'column' : 'row', flex: 1, minHeight: 0, gap: isMobile ? '10px' : '20px' }}>
          
          {/* 📊 통계 탭 영역 (activeTab이 stats일 때만 보임) */}
          <div style={{ 
            display: activeTab === "stats" ? 'flex' : 'none',
            flex: 1, height: '100%', overflowY: 'auto', flexDirection: 'column'
          }}>
            <StatsPanel isMobile={isMobile} host={host} />
          </div>

          {/* 📡 관제/시뮬레이션 탭 영역 (stats가 아닐 때 보임) */}
          <div style={{ 
            display: activeTab !== "stats" ? 'flex' : 'none',
            flex: 1, flexDirection: isMobile ? 'column' : 'row', gap: isMobile ? '10px' : '20px', minHeight: 0
          }}>
            {/* 왼쪽 큰 섹션: 비디오(상) + 지도(하) */}
            <div style={{ flex: 3.2, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              
              {/* 📹 비디오 패널: 지도가 줄어들면 자동으로 늘어나도록 설정 */}
              <div style={{
                ...panelWrapper, 
                flex: 1, // ✨ 핵심: 항상 1을 유지하여 지도가 사라지면 남은 공간을 다 차지함
                marginBottom: (isEmergency && showMap) ? (isMobile ? '10px' : '20px') : '0px',
                transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)' 
              }}>
                <VideoPanel videoUrl={videoUrl} activeTab={activeTab} cctvData={cctvData} host={host} user={user}/>
              </div>

              {/* 📍 지도 펼치기 버튼: 지도가 숨겨졌을 때만 비디오 하단에 나타남 */}
              {isEmergency && !showMap && (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '10px 0' }}>
                  <button 
                    onClick={() => setShowMap(true)} 
                    style={reShowBtnStyle}
                  >
                    📍 지도 펼치기 (위치 확인)
                  </button>
                </div>
              )}

              {/* 🗺️ 지도 패널: maxHeight를 조절하여 '스르륵' 효과 구현 */}
              <div style={{
                ...panelWrapper, 
                // ✨ display: none을 쓰지 않고 스타일로 높이를 0으로 만듭니다.
                maxHeight: (isEmergency && showMap) ? (isMobile ? '400px' : '600px') : '0px',
                opacity: (isEmergency && showMap) ? 1 : 0,
                flex: (isEmergency && showMap) ? 1.5 : 0, 
                overflow: 'hidden',
                transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                border: (isEmergency && showMap) ? '1px solid #1e293b' : '0px solid transparent'
              }}>
                <MapPanel 
                  activeTab={activeTab} 
                  isEmergency={isEmergency} 
                  mapRef={mapRef} 
                  onHide={() => setShowMap(false)} 
                />
              </div>
            </div>

            <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', gap: isMobile ? '10px' : '20px', minHeight: 0 }}>
              <div style={{ flex: 1, minHeight: isMobile ? '300px' : 0 }}>
                  <ControlPanel activeTab={activeTab} startSim={startSim} logs={logs} />
              </div>
              <div style={{ 
                ...alertListPanel, 
                flex: 0.72, 
                minHeight: isMobile ? '250px' : 0, 
                display: 'flex', 
                flexDirection: 'column',
                overflow: 'hidden'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexShrink: 0 }}>
                  <h4 style={{ color: '#f87171', fontSize: '14px', margin: 0 }}>🚨 미조치 알림 ({pendingAlerts.length})</h4>
                  {pendingAlerts.length > 0 && (
                    <button onClick={resolveAllAlerts} style={resolveAllBtn}>전체 조치</button>
                  )}
                </div>
                <div style={{ flex: 1, overflowY: 'auto', paddingRight: '5px' }}>
                  {pendingAlerts.length === 0 ? (
                    <div style={{ color: '#475569', fontSize: '12px', textAlign: 'center', padding: '10px' }}>현재 상황 없음</div>
                  ) : (
                    pendingAlerts.map(alert => (
                      <div key={alert.id} style={miniAlertCard}>
                        <div onClick={() => handleAlertClick(alert)} style={{ flex: 1, cursor: 'pointer' }}>
                          <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#f87171' }}>{alert.type}</div>
                          <div style={{ fontSize: '11px', color: '#94a3b8' }}>{alert.address}</div>
                        </div>
                        <button onClick={() => resolveEmergency(alert.id, alert.type, alert.address, alert.origin, user?.name, activeTab === "sim")} style={resolveBtn}>조치</button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

const pulseAnimation = `@keyframes emergency-bg { 0%, 100% { background-color: #020617; } 50% { background-color: #1e0000; } }`;
const hideScrollbar = `*::-webkit-scrollbar { display: none !important; } * { -ms-overflow-style: none !important; scrollbar-width: none !important; }`;
const containerStyle = { display: 'flex', background: '#020617', color: '#fff', width: '100vw', boxSizing: 'border-box' };
const mainWrapper = { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, boxSizing: 'border-box' };
const contentHeader = { display: 'flex', justifyContent: 'space-between', alignItems: 'center' };
const headerTitle = { fontWeight: '800' };
const adminInfo = { color: '#64748b', fontSize: '13px', marginRight: '20px' };
const gridContainer = { display: 'flex' };
const panelWrapper = { background: '#0f172a', borderRadius: '12px', border: '1px solid #1e293b', overflow: 'hidden', display: 'flex', flexDirection: 'column', boxSizing: 'border-box' };
const alertListPanel = { background: '#0f172a', padding: '15px', borderRadius: '12px', border: '1px solid #451a1a', boxSizing: 'border-box' };
const miniAlertCard = { display: 'flex', alignItems: 'center', padding: '10px', background: '#1e293b', borderRadius: '8px', border: '1px solid #334155', marginBottom: '8px' };
const resolveBtn = { background: '#2563eb', color: 'white', border: 'none', padding: '5px 12px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold', cursor: 'pointer' };
const resolveAllBtn = { background: 'transparent', color: '#f87171', border: '1px solid #f87171', padding: '3px 8px', borderRadius: '4px', fontSize: '11px', cursor: 'pointer' };
const reShowBtnStyle = {
  background: '#1e293b',
  color: '#38bdf8',
  border: '1px solid #38bdf8',
  padding: '8px 16px',
  borderRadius: '20px',
  fontSize: '13px',
  fontWeight: 'bold',
  cursor: 'pointer',
  boxShadow: '0 0 10px rgba(56, 189, 248, 0.2)',
  transition: 'all 0.2s'
};
export default Dashboard;