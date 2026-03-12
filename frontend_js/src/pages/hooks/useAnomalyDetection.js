/* eslint-disable */
import { useState, useEffect, useRef } from 'react';
import Swal from 'sweetalert2';
import axios from 'axios';
import { useMapMarkers } from './useMapMarkers';

export function useAnomalyDetection(socket, activeTab, setActiveTab, setVideoUrl, host, adminName) {
  const [isEmergency, setIsEmergency] = useState(false);
  const [pendingAlerts, setPendingAlerts] = useState([]);
  const [logs, setLogs] = useState([]);
  const mapRef = useRef(null);
  const pendingAlertsRef = useRef([]);

  // 마커 관리 훅 연결
  const { markersRef, createMarker, removeMarker } = useMapMarkers(mapRef);

  useEffect(() => { pendingAlertsRef.current = pendingAlerts; }, [pendingAlerts]);

  // ✅ 초기 미조치 데이터 불러오기
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const response = await axios.get(`http://${host}:5000/api/pending_alerts`);
        if (response.data.length > 0) {
          setPendingAlerts(response.data);
          setIsEmergency(true);
        }
      } catch (err) { console.error("❌ 데이터 로드 실패:", err); }
    };
    fetchInitialData();
  }, [host]);

  // ✅ 알림 목록 변경 시 마커 그리기
  useEffect(() => {
    pendingAlerts.forEach(alert => createMarker(alert, () => resolveEmergency(alert.id, alert.type, alert.address, alert.origin, alert.isSimulation)));
  }, [pendingAlerts]);

  const resolveEmergency = async (alertId, type, address, originType, isSimulation = false) => {
    try {
      const result = await Swal.fire({
        title: '조치 및 상황 확인',
        text: `[${type}] 상황이 실제 상황입니까?`,
        icon: 'question',
        showCancelButton: true,
        confirmButtonText: '✅ 실제상황 (정탐)',
        cancelButtonText: '❌ 알람오류 (오탐)',
        confirmButtonColor: '#2563eb',
        cancelButtonColor: '#f87171',
        reverseButtons: true
      });

      const isCorrect = result.isConfirmed ? 1 : 0;

      // DB 업데이트 (단일 건)
      await axios.post(`http://${host}:5000/api/resolve_alert_db`, { 
        alertId, isCorrect, adminName: adminName, is_simulation: isSimulation ? 1 : 0 
      });

      // UI/마커 제거
      removeMarker(alertId);
      setPendingAlerts(prev => {
        const updated = prev.filter(a => String(a.id) !== String(alertId));
        if (updated.length === 0) setIsEmergency(false);
        return updated;
      });

      // 로그 및 소켓 전파
      const statusLabel = isCorrect ? "정탐" : "오탐";
      setLogs(prev => [`[${new Date().toLocaleTimeString()}] ✅ ${statusLabel} 조치: ${type}`, ...prev]);
      if (socket) {
        socket.emit("resolve_emergency", { alertId, type, address, isCorrect, adminName: adminName, isSimulation, senderId: socket.id });
      }
      Swal.fire({ title: `${statusLabel} 완료`, icon: 'success', timer: 1000, showConfirmButton: false });
    } catch (err) { console.error(err); }
  };
// ✅ 상황 조치 완료 실행 함수 (정탐/오탐 피드백 추가)
  const resolveAllAlertsAction = async (alerts, isSimulation = false) => {
    const result = await Swal.fire({
      title: '일괄 조치',
      text: `${alerts.length}건을 모두 정탐 처리하시겠습니까?`,
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: '네, 모두 처리',
      cancelButtonText: '취소'
    });

    if (result.isConfirmed) {
      try {
        const alertIds = alerts.map(a => a.id);
        const isSim = alerts[0]?.isSimulation || false;
        // ✅ [변경] 개별 루프 대신 벌크 API 딱 한 번 호출
        await axios.post(`http://${host}:5000/api/resolve_alerts_bulk`, { 
          alertIds: alertIds, 
          isCorrect: 1, 
          adminName: adminName,
          is_simulation: isSim ? 1 : 0 
        });

        // 내 화면 마커들 제거
        alertIds.forEach(id => removeMarker(id));
        setPendingAlerts([]);
        setIsEmergency(false);

        // 타 관리자에게 전파 (소켓은 가벼우므로 루프 돌려도 무방)
        if (socket) {
          alerts.forEach(alert => {
            socket.emit("resolve_emergency", { 
              alertId: alert.id, 
              type: alert.type, 
              isCorrect: 1, 
              isSimulation, 
              senderId: socket.id 
            });
          });
        }

        // 상태 최종 업데이트
        setPendingAlerts([]);
        setIsEmergency(false);
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] 📦 일괄 조치 완료 (${alerts.length}건)`, ...prev]);
        
        Swal.fire('일괄 처리 완료', '', 'success');

      } catch (err) {
        console.error("일괄 처리 중 에러:", err);
        Swal.fire('처리 실패', 'DB 업데이트 중 오류가 발생했습니다.', 'error');
      }
    }
  };

  // ✅ 소켓 이벤트 리스너
  useEffect(() => {
    if (!socket) return;

    // 1. 영상 동기화
    const handleForceStart = (data) => {
      setActiveTab("sim");
      const syncUrl = `http://${host}:5000/api/video_feed?type=${data.type}&v=${Date.now()}`;
      setVideoUrl(syncUrl);
      if (data.lat && mapRef.current) mapRef.current.panTo(new window.kakao.maps.LatLng(data.lat, data.lng));
    };

    // 2. 새로운 이상감지 발생
    const handleAnomaly = (data) => {
      if (!window.kakao || !window.kakao.maps) {
        console.warn("⚠️ 카카오 맵 API가 아직 로드되지 않았습니다.");
        return;
      }
      if (pendingAlertsRef.current.some(a => String(a.id) === String(data.alert_id))) return;
  
      const time = new Date().toLocaleTimeString();
      let coord;
      try {
        coord = new window.kakao.maps.LatLng(data.lat, data.lng);
      } catch (e) {
        console.error("❌ 좌표 생성 실패:", e);
        return;
      }

  // 💡 [핵심 추가] 실제 ITS 데이터인지 확인 (is_simulation이 False이거나 origin이 실제인 경우)
  // 서버에서 보낸 data.video_origin이 'realtime_its'인 경우 주소 변환을 건너뜁니다.

  // ✅ 서버에서 준 is_simulation이 있으면 사용, 없으면 문자열로 판별
  const isSimulationValue = data.hasOwnProperty('is_simulation') 
    ? data.is_simulation 
    : data.video_origin !== 'realtime_its';

  // 실제 데이터 여부 확인 (주소 업데이트 방지용)
  const isRealtime = !isSimulationValue;

  const processAlert = (finalAddress) => {
    // 1️⃣ 실제 데이터가 아닐 때(시뮬레이션일 때)만 DB 주소 업데이트 실행
      if (!isRealtime) {
        axios.post(`http://${host}:5000/api/update_address`, { 
          alertId: data.alert_id, 
          address: finalAddress 
        });
      } else {
        console.log("🛡️ [보호] 실제 CCTV 데이터이므로 주소 업데이트를 수행하지 않습니다:", finalAddress);
      }

      // 2️⃣ 리스트에 추가
      const newAlert = { 
        id: data.alert_id, 
        type: data.type, 
        address: finalAddress, 
        time, 
        origin: data.video_origin, 
        isSimulation: isSimulationValue,
        lat: data.lat, 
        lng: data.lng,
        imageUrl: data.image_url 
      };

      setPendingAlerts(prev => [newAlert, ...prev]);
      setIsEmergency(true);
      setLogs(prev => [`[${time}] 🚨 감지: ${data.type}`, ...prev]);

      if (mapRef.current && typeof mapRef.current.panTo === 'function') {
        mapRef.current.panTo(coord);
      }

      // 3️⃣ 지도 이동 및 팝업
      mapRef.current?.panTo(coord);
      Swal.fire({
        title: `🚨 ${data.type} 감지`,
        html: `위치: ${finalAddress}`,
        icon: 'error',
        timer: 2000
      });
    };

    if (isRealtime) {
      // 🅰️ 실제 데이터면: 서버에서 보내준 address(CCTV 이름)를 그대로 사용
      processAlert(data.address || data.video_origin);
    } else {
      if (!window.kakao.maps.services || !window.kakao.maps.services.Geocoder) {
        processAlert(data.video_origin || "위치 정보 확인 불가");
        return;
      }

      // 🅱️ 시뮬레이션이면: 기존처럼 카카오 API로 주소 변환
      const geocoder = new window.kakao.maps.services.Geocoder();
      geocoder.coord2Address(coord.getLng(), coord.getLat(), (result, status) => {
        const convertedAddress = status === window.kakao.maps.services.Status.OK 
          ? result[0].address.address_name 
          : (data.video_origin || "위치 정보 확인 불가");
        
        processAlert(convertedAddress);
      });
    }
  };

    // 3. 타 관리자의 조치 완료 수신
    const handleRemoteResolve = (data) => {
  // 내가 보낸 신호라면 무시
  if (data.senderId === socket.id) return;

  // 마커 제거 및 상태 업데이트
  removeMarker(data.alertId);
  setPendingAlerts(prev => {
    const updated = prev.filter(a => String(a.id) !== String(data.alertId));
    if (updated.length === 0) setIsEmergency(false);
    return updated;
  });

  // ✅ 타 관리자 처리 알림 추가
    const statusLabel = data.isCorrect ? "실제상황 정탐" : "알람오류 오탐";
    
      Swal.fire({
        title: '타 관리자 조치 완료',
        text: `[${data.type}] 상황을 다른 관리자가 ${statusLabel}으로 처리하였습니다.`,
        icon: 'info',
        toast: true,           // 토스트 형식으로 작게 표시 (방해 금지)
        position: 'top-end',   // 우측 상단에 표시
        showConfirmButton: false,
        timer: 3000,           // 3초간 표시
        timerProgressBar: true,
        width: '600px'
      });

      // 로그.
      setLogs(prev => [`[${new Date().toLocaleTimeString()}] ℹ️ 타 관리자 조치: ${data.type}`, ...prev]);
    };

    socket.on("force_video_start", handleForceStart);
    socket.on("anomaly_detected", handleAnomaly);
    socket.on("emergency_resolved", handleRemoteResolve);

    return () => {
      socket.off("force_video_start", handleForceStart);
      socket.off("anomaly_detected", handleAnomaly);
      socket.off("emergency_resolved", handleRemoteResolve);
    };
  }, [socket, host]);

  return { isEmergency, pendingAlerts, logs, mapRef, resolveEmergency, resolveAllAlertsAction, moveToAlert: (a) => mapRef.current?.panTo(new window.kakao.maps.LatLng(a.lat, a.lng)) };
}