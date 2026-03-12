/* eslint-disable */
import { useRef } from 'react';

export function useMapMarkers(mapRef) {
  const markersRef = useRef({});

  // ✅ 마커 생성 함수
  const createMarker = (alert, resolveEmergency) => {
    if (!mapRef.current || markersRef.current[alert.id] || !window.kakao) return;

    const coord = new window.kakao.maps.LatLng(alert.lat, alert.lng);
    
    // 버튼 클릭 시 호출될 전역 함수 연결
    window.resolveFromMap = (id, type, addr, origin) => resolveEmergency(id, type, addr, origin);

    const content = `
      <div style="position: relative; bottom: 50px; background: white; border-radius: 12px; border: 2px solid #ff4d4f; box-shadow: 0 4px 12px rgba(0,0,0,0.2); padding: 10px; min-width: 140px;">
        <div style="display:flex; flex-direction:column; align-items:center; gap:5px;">
          <span style="font-size:12px; font-weight:800; color:#ff4d4f; white-space:nowrap;">⚠️ ${alert.type}</span>
          <span style="font-size:11px; color:#666; text-align:center; display:block; width:100%; word-break:break-all;">
            ${alert.address || "주소 확인 중..."}
          </span>
          <button onclick="window.resolveFromMap('${alert.id}', '${alert.type}', '${alert.address}', '${alert.origin}')" 
                  style="background:#ff4d4f; color:white; border:none; padding:4px 10px; border-radius:6px; font-size:11px; cursor:pointer; font-weight:bold; margin-top:5px;">
            조치 완료
          </button>
        </div>
        <div style="position: absolute; bottom: -10px; left: 50%; transform: translateX(-50%); width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 10px solid #ff4d4f;"></div>
      </div>`;

    const overlay = new window.kakao.maps.CustomOverlay({
      content,
      map: mapRef.current,
      position: coord,
      yAnchor: 1 
    });

    markersRef.current[alert.id] = overlay;
  };

  // ✅ 마커 제거 함수
  const removeMarker = (alertId) => {
    if (markersRef.current[alertId]) {
      markersRef.current[alertId].setMap(null);
      delete markersRef.current[alertId];
    }
  };

  return { markersRef, createMarker, removeMarker };
}