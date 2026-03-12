/* eslint-disable */
import React, { useEffect } from 'react';

const MapPanel = ({ activeTab, isEmergency, mapRef, onHide }) => {
  useEffect(() => {
    const initMap = () => {
      const container = document.getElementById('map');
      if (!container) return;

      if (mapRef.current) {
        mapRef.current.relayout();
        // ✨ 지도가 다시 나타날 때 현재 중심 좌표를 유지하거나 다시 설정
        const currentCenter = mapRef.current.getCenter();
        mapRef.current.setCenter(currentCenter);
        return;
      }

      const options = { 
        center: new window.kakao.maps.LatLng(37.5665, 126.9780), 
        level: 4 
      };
      mapRef.current = new window.kakao.maps.Map(container, options);
    };

    if (window.kakao && window.kakao.maps) {
      window.kakao.maps.load(initMap);
    }

    const handleResize = () => {
      if (mapRef.current) mapRef.current.relayout();
    };

    window.addEventListener('resize', handleResize);
    
    // 🛠️ 애니메이션 보정을 위한 인터벌 설정
    const animationInterval = setInterval(() => {
      if (mapRef.current) {
        mapRef.current.relayout();
        mapRef.current.setCenter(mapRef.current.getCenter());
      }
    }, 50);

    // 0.6초 뒤 인터벌을 멈추기 위한 타이머
    const stopTimer = setTimeout(() => {
      clearInterval(animationInterval);
    }, 600);

    return () => {
      window.removeEventListener('resize', handleResize);
      clearInterval(animationInterval); // timers 대신 이거!
      clearTimeout(stopTimer);          // 이것도 추가!
    };
  }, [activeTab, isEmergency]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{
        padding: '12px 20px', 
        background: isEmergency ? '#7f1d1d' : '#111827',
        borderBottom: '1px solid #1e293b', 
        color: '#fff', 
        fontSize: '14px',
        fontWeight: 'bold', 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        flexShrink: 0,
        transition: 'background 0.3s ease'
      }}>
        <span>🗺️ 현장 지도 {isEmergency && "(긴급 상황 발생)"}</span>
        {isEmergency && (
          <button 
            onClick={onHide} 
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              color: '#fff',
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            ✕ 지도 숨기기
          </button>
        )}
      </div>
      <div id="map" style={{ width: '100%', flex: 1, minHeight: '300px', background: '#0f172a' }}></div>
    </div>
  );
};

export default MapPanel;