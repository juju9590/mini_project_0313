/* eslint-disable */
import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import axios from 'axios';
import Swal from 'sweetalert2';

// 개별 비디오/이미지를 렌더링하는 내부 컴포넌트
const SingleMedia = ({ url, isHls, name, isFlashing }) => {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);

  useEffect(() => {
    if (isHls && url && videoRef.current) {
      if (hlsRef.current) hlsRef.current.destroy();

      if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
        videoRef.current.src = url;
      } else if (Hls.isSupported()) {
        const hls = new Hls();
        hls.loadSource(url);
        hls.attachMedia(videoRef.current);
        hlsRef.current = hls;
      }
    }
    return () => { if (hlsRef.current) hlsRef.current.destroy(); };
  }, [url, isHls]);

  return (
    <div style={gridItemStyle}>
      {isHls ? (
        <video ref={videoRef} autoPlay muted playsInline style={mediaStyle} />
      ) : (
        <img 
          src={url} 
          style={{ 
            ...mediaStyle, 
            filter: isFlashing ? 'brightness(2.5)' : 'none', // 캡처 시 밝게 반짝
            transition: 'filter 0.15s ease' 
          }} 
          alt="streaming" 
        />
      )}
      <div style={miniLabelStyle}>{name}</div>
    </div>
  );
};

// 메인 컴포넌트, 대시보드 중앙의 영상 영역 관리(단일 피드, 4분할 cctv, 확대 모달, 캡처 기능)
const VideoPanel = ({ videoUrl, activeTab, cctvData = [], host, user }) => {
  const isCctvMode = activeTab === "cctv";
  const [isFlashing, setIsFlashing] = useState(false);
  const [expandedMedia, setExpandedMedia] = useState(null);
  
  // 📸 수동 캡처 로직 (메모 입력 포함)
  const handleCapture = async () => {
    console.log("📸 버튼 클릭됨, 함수 시작!");
    const targetType = activeTab === 'cctv' ? 'fire' : activeTab;
  try {
    // 1️⃣ [즉시 실행] 버튼 누르자마자 서버에 "지금" 프레임을 찍으라고 명령
    const res = await axios.post(`http://${host}:5000/api/capture_now`, {
      type: targetType, // 'sim' 또는 'webcam'
      adminName: user?.name || '관리자'
    });

    const { db_id, image_url } = res.data;

    // 2️⃣ [그 다음] 메모를 물어봄 (이미 사진은 위에서 찍혔으므로 천천히 입력해도 됨)
    const { value: text, isConfirmed } = await Swal.fire({
      title: '장면 캡처 완료',
      text: '기록할 메모가 있나요?',
      input: 'text',
      imageUrl: `http://${host}:5000${image_url}`, // 방금 찍은 사진 보여주기
      imageWidth: 300,
      showCancelButton: true,
      confirmButtonText: '메모 저장',
      cancelButtonText: '사진만 저장'
    });

    // 3️⃣ 사용자가 메모를 쓰고 '확인'을 눌렀을 때만 추가 저장
    if (isConfirmed && text) {
      await axios.post(`http://${host}:5000/api/update_capture_memo`, {
        db_id: db_id,
        memo: text
      });
    }
  } catch (err) {
    console.error("캡처 오류:", err);
  }
};

  return (
    <div style={containerStyle}>
      {/* 상단 레이블 */}
      <div style={labelStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '6px', height: '6px', background: '#ef4444', borderRadius: '50%' }} /> 
          {isCctvMode ? "고속도로 실시간 멀티 관제" : "LOCAL FEED"}
        </div>
        <div style={{ fontSize: '10px', opacity: 0.6 }}>{activeTab.toUpperCase()} MODE</div>
      </div>

      {isCctvMode ? (
        <div style={quadGridStyle}>
          {cctvData.length > 0 ? (
            cctvData.slice(0,4).map((item, idx) => {
              // 🔥 1. 첫 번째 화면(idx === 0)만 분석 모드 적용
              const isReverseAnalysis = idx === 0; 
              const isFireAnalysis = idx === 1;

              let finalUrl = item.url;
              let isHls = true; // 기본 HLS 재생

              if (isReverseAnalysis) {
                finalUrl =
                  `http://${host}:5000/api/its/video_feed?mode=reverse` +
                  `&url=${encodeURIComponent(item.url)}` +
                  `&name=${encodeURIComponent(item.name)}` +
                  `&lat=${item.lat}&lng=${item.lng}`;

                isHls = false;
              } 
              else if (isFireAnalysis) {
                finalUrl =
                  `http://${host}:5000/api/its/video_feed?mode=fire` +
                  `&url=${encodeURIComponent(item.url)}` +
                  `&name=${encodeURIComponent(item.name)}` +
                  `&lat=${item.lat}&lng=${item.lng}`;

                isHls = false;
              }

              let displayName = item.name;
              if (isReverseAnalysis) displayName = `🔴 [AI 역주행 분석] ${item.name}`;
              if (isFireAnalysis) displayName = `🔥 [AI 화재 분석] ${item.name}`;

              return (
                <div 
                  key={idx} 
                  style={{ cursor: 'zoom-in', width: '100%', height: '100%' }}
                  onClick={() => setExpandedMedia({ url: finalUrl, isHls: isHls, name: item.name })}
                >
                  <SingleMedia 
                    url={finalUrl} 
                    isHls={isHls} 
                    name={displayName} 
                  />
                </div>
              );
            })
          ) : (
            <div style={loadingStyle}>CCTV 데이터를 불러오는 중...</div>
          )}
        </div>
      ) : (
        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
          {videoUrl ? (
            <>
              <SingleMedia url={videoUrl} isHls={false} isFlashing={isFlashing} />
              {/* 📸 수동 캡처 버튼 (단일 모드에서만 노출) */}
              <button 
                onClick={handleCapture}
                style={captureBtnStyle}
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.1)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                title="현재 화면 수동 기록 및 메모"
              >
                📷
              </button>
            </>
          ) : (
            <div style={loadingStyle}>연결 신호 대기 중...</div>
          )}
        </div>
      )}
      {expandedMedia && (
        <div style={modalOverlayStyle} onClick={() => setExpandedMedia(null)}>
          <div style={modalContentStyle} onClick={(e) => e.stopPropagation()}>
            {/* 아래 div가 실제 영상의 '틀'이 됩니다. 
                aspectRatio를 주면 영상 비율(16:9)을 유지하며 꽉 차게 만들기 좋습니다.
            */}
            <div style={{ 
              position: 'relative', 
              width: '95vw',      // 너비 90%
              height: '90vh',     // 높이 80%
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#000',
              borderRadius: '8px',
              overflow: 'hidden'
            }}>
              <div style={{ width: '100%', height: '100%' }}>
                <SingleMedia 
                  url={expandedMedia.url} 
                  isHls={expandedMedia.isHls} 
                  name={expandedMedia.name} 
                />
              </div>
              
              {/* 우측 상단 닫기 버튼을 영상 위에 띄움 */}
              <button 
                style={closeBtnStyle} 
                onClick={() => setExpandedMedia(null)}
              >
                ✕ 닫기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// --- 스타일 정의 ---

const modalOverlayStyle = {
  position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
  backgroundColor: 'rgba(0,0,0,0.85)', zIndex: 9999, // 대시보드 사이드바보다 위에 뜨도록
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  backdropFilter: 'blur(5px)'
};

const modalContentStyle = {
  padding: '5px',
  background: '#1e293b',
  borderRadius: '12px',
  // 🔥 Shorthand 'border' 대신 개별 속성 사용 (경고 해결)
  borderWidth: '2px',
  borderStyle: 'solid',
  borderColor: '#334155', 
  overflow: 'hidden',
  display: 'flex', // 내부 틀이 꽉 차도록 추가
  alignItems: 'center',
  justifyContent: 'center'
};

const closeBtnStyle = {
  position: 'absolute', top: '10px', right: '10px',
  background: 'rgba(239, 68, 68, 0.8)', color: 'white', border: 'none',
  padding: '8px 15px', borderRadius: '6px', cursor: 'pointer',
  fontSize: '14px', fontWeight: 'bold', zIndex: 10
};

const containerStyle = { 
  height: '100%', background: '#000', position: 'relative', overflow: 'hidden' 
};

const quadGridStyle = {
  display: 'grid',
  // ✨ minmax(0, 1fr)를 써야 내부 영상이 커도 틀이 늘어나지 않습니다.
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', 
  gridTemplateRows: 'repeat(2, minmax(0, 1fr))', 
  width: '100%',
  height: '100%',
  gap: '4px',
  background: '#020617',
  boxSizing: 'border-box',
  overflow: 'hidden' // 중요: 자식 요소가 튀어나오지 못하게 함
};

const gridItemStyle = {
  position: 'relative',
  width: '100%',
  height: '100%',
  background: '#000',
  display: 'flex',
  alignItems: 'center',      // 중앙 정렬 (여백 발생 시)
  justifyContent: 'center',
  overflow: 'hidden',
  border: '1px solid #1e293b'
};

const mediaStyle = { 
  // ✨ 사용자의 아이디어: 영상이 칸의 90%만 차지하게 함
  maxWidth: '95%', 
  maxHeight: '95%', 
  width: 'auto',      
  height: 'auto',     
  objectFit: 'contain', 
  display: 'block',
  
  // 시각적 팁: 영상에 미세한 그림자를 주면 공중에 떠 있는 느낌이 납니다.
  boxShadow: '0 0 15px rgba(0,0,0,0.5)', 
  borderRadius: '2px' // 영상 모서리도 아주 살짝 굴리면 부드러워요
};

const miniLabelStyle = {
  position: 'absolute', 
  bottom: '12px', // 여백 고려하여 살짝 올림
  left: '12px',
  background: 'rgba(15, 23, 42, 0.8)', // 조금 더 불투명하게
  color: '#fff',
  padding: '4px 10px', 
  fontSize: '11px', 
  borderRadius: '4px',
  border: '1px solid rgba(99, 102, 241, 0.3)', // 테두리에 강조색 살짝 가미
  pointerEvents: 'none',
  zIndex: 5
};

const labelStyle = { 
  position: 'absolute', top: 0, left: 0, right: 0, padding: '10px 15px',
  background: 'linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%)', 
  zIndex: 2, display: 'flex', justifyContent: 'space-between',
  color: '#fff', fontSize: '12px', fontWeight: 'bold', pointerEvents: 'none'
};

const captureBtnStyle = {
  position: 'absolute',
  right: '20px',
  bottom: '20px',
  width: '50px',
  height: '50px',
  borderRadius: '50%',
  backgroundColor: 'rgba(37, 99, 235, 0.8)',
  border: '2px solid rgba(255, 255, 255, 0.3)',
  color: 'white',
  fontSize: '22px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backdropFilter: 'blur(4px)',
  transition: 'all 0.2s ease',
  zIndex: 10,
  boxShadow: '0 4px 15px rgba(0,0,0,0.5)'
};

const loadingStyle = { color: '#475569', fontSize: '13px' };

export default VideoPanel;