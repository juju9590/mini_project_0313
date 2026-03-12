/* eslint-disable */
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function StatsPanel({ isMobile, host }) {
  const [viewMode, setViewMode] = useState("summary"); // "summary" | "history"
  const [dataMode, setDataMode] = useState("real");   // ✅ 추가: "real" | "sim" | "all"
  const [stats, setStats] = useState({
    total: 0, correct: 0, incorrect: 0, precision: 0,
    typeData: { fire: 0, reverse: 0, manual: 0 } 
  });
  const [historyLogs, setHistoryLogs] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  // 1. 통계 요약 데이터 로드
  useEffect(() => {
    const fetchStats = async () => {
      try {
        // ✅ API 호출 시 dataMode를 쿼리 스트링으로 전달
        const response = await axios.get(`http://${host}:5000/api/stats/summary?mode=${dataMode}`);
        setStats({
          total: response.data.total,
          correct: response.data.correct,
          incorrect: response.data.incorrect,
          precision: response.data.precision,
          typeData: response.data.type_counts || { fire: 0, reverse: 0, manual: 0 }
        });
      } catch (err) { 
        console.error("통계 로드 실패:", err); 
      }
    };

    fetchStats();
    if (viewMode === "summary") {
      const interval = setInterval(fetchStats, 5000); // 5초마다 갱신
      return () => clearInterval(interval);
    }
  }, [host, viewMode, dataMode]); // ✅ dataMode가 바뀔 때마다 다시 로드

  // 2. 상세 이력 데이터 로드
  useEffect(() => {
    if (viewMode === "history") {
      const fetchHistory = async () => {
        try {
          // ✅ 이력 데이터도 dataMode 파라미터 추가
          const response = await axios.get(`http://${host}:5000/api/stats/history?date=${selectedDate}&mode=${dataMode}`);
          setHistoryLogs(response.data.logs);
        } catch (err) { 
          console.error("이력 로드 실패:", err); 
        }
      };
      fetchHistory();
    }
  }, [viewMode, selectedDate, host, dataMode]); // ✅ dataMode가 바뀔 때마다 다시 로드

  const calculatePercent = (count) => {
    const sum = stats.typeData.fire + stats.typeData.reverse + stats.typeData.manual;
    return sum > 0 ? (count / sum) * 100 : 0;
  };

  return (
    <div style={mainContainerStyle}>
      {/* 🚀 상단 헤더 */}
      <header style={{ marginBottom: '20px', flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '15px' }}>
          <div>
            <h3 style={{ color: '#fff', fontSize: '22px', fontWeight: '700', margin: '0 0 5px 0' }}>
              📊 데이터 분석 센터
            </h3>
            
            {/* ✅ 데이터 모드 스위칭 버튼 그룹 */}
            <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
              <button 
                onClick={() => setDataMode("real")} 
                style={dataMode === "real" ? activeDataBtn : inactiveDataBtn}
              >실제 데이터</button>
              <button 
                onClick={() => setDataMode("sim")} 
                style={dataMode === "sim" ? activeDataBtn : inactiveDataBtn}
              >시뮬레이션</button>
              <button 
                onClick={() => setDataMode("all")} 
                style={dataMode === "all" ? activeDataBtn : inactiveDataBtn}
              >전체 보기</button>
            </div>
          </div>
          
          <div style={tabGroupStyle}>
            <button 
              onClick={() => setViewMode("summary")} 
              style={viewMode === "summary" ? activeTabBtn : inactiveTabBtn}
            >통계 요약</button>
            <button 
              onClick={() => setViewMode("history")} 
              style={viewMode === "history" ? activeTabBtn : inactiveTabBtn}
            >상세 이력</button>
          </div>
        </div>
      </header>

      {/* --- [Tab 1] 통계 요약 --- */}
      {viewMode === "summary" ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* 상단 4종 카드 */}
          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr 1fr' : '1fr 1fr 1fr 1fr', gap: '15px' }}>
            <div style={statCardStyle}><div style={statTitleStyle}>전체 건수 ({dataMode === 'real' ? '실제' : dataMode === 'sim' ? '시뮬' : '통합'})</div><div style={statValueStyle}>{stats.total}</div></div>
            <div style={{ ...statCardStyle, borderLeft: '4px solid #22c55e' }}><div style={statTitleStyle}>정탐</div><div style={{ ...statValueStyle, color: '#22c55e' }}>{stats.correct}</div></div>
            <div style={{ ...statCardStyle, borderLeft: '4px solid #f87171' }}><div style={statTitleStyle}>오탐</div><div style={{ ...statValueStyle, color: '#f87171' }}>{stats.incorrect}</div></div>
            <div style={{ ...statCardStyle, borderLeft: '4px solid #6366f1' }}><div style={statTitleStyle}>신뢰도</div><div style={{ ...statValueStyle, color: '#6366f1' }}>{stats.precision}%</div></div>
          </div>

          {/* 중앙 차트 및 분포 */}
          <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '20px' }}>
            <div style={{ ...panelWrapper, flex: 1, padding: '20px' }}>
              <h4 style={chartTitleStyle}>🔥 이벤트 분포 ({dataMode.toUpperCase()})</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '10px' }}>
                {['fire', 'reverse', 'manual'].map(type => (
                  <div key={type}>
                    <div style={labelRowStyle}>
                      <span style={{color: '#cbd5e1'}}>{type === 'fire' ? '화재' : type === 'reverse' ? '역주행' : '수동기록'}</span>
                      <span style={{fontWeight: 'bold', color: type === 'fire' ? '#ef4444' : type === 'reverse' ? '#f59e0b' : '#3b82f6'}}>{stats.typeData[type]}건</span>
                    </div>
                    <div style={barBackgroundStyle}>
                      <div style={{ ...barProgressStyle, width: `${calculatePercent(stats.typeData[type])}%`, background: type === 'fire' ? '#ef4444' : type === 'reverse' ? '#f59e0b' : '#3b82f6' }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ ...panelWrapper, flex: 1.5, padding: '20px' }}>
              <h4 style={chartTitleStyle}>📈 시간대별 이상감지 추이</h4>
              <div style={chartPlaceholderStyle}>데이터 연동 대기 중</div>
            </div>
          </div>

          {/* 💡 하단 인사이트 섹션 */}
          <div style={{ 
            ...panelWrapper, 
            padding: '25px', 
            border: '1px solid #1e293b', 
            background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
            marginTop: '10px'
          }}>
            <h4 style={{ ...chartTitleStyle, color: '#3b82f6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{fontSize: '20px'}}>💡</span> Model Feedback Insight
            </h4>
            <p style={{ color: '#94a3b8', fontSize: '15px', lineHeight: '1.8', margin: 0, wordBreak: 'keep-all' }}>
              현재 <b>{dataMode === 'real' ? "실제 운영 데이터" : dataMode === 'sim' ? "시뮬레이션 데이터" : "전체 누적 데이터"}</b>를 분석 중입니다. 
              관리자의 조치 결과와 {stats.typeData.manual}건의 수동 기록은 모델 학습 파이프라인의 핵심 피드백 데이터로 활용됩니다.
            </p>
          </div>
        </div>
      ) : (
        /* --- [Tab 2] 상세 이력 --- */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <div style={filterBar}>
              <span style={{ color: '#94a3b8', fontSize: '14px' }}>날짜 선택:</span>
              <input 
                type="date" 
                value={selectedDate} 
                onChange={(e) => setSelectedDate(e.target.value)} 
                style={datePickerStyle} 
              />
              <span style={{ color: '#475569', fontSize: '12px', marginLeft: '10px' }}>필터: {dataMode === 'real' ? '실제상황만' : dataMode === 'sim' ? '시뮬레이션만' : '전체 기록'}</span>
          </div>

          <div style={logListContainer}>
            {historyLogs.length === 0 ? (
              <div style={emptyStyle}>해당 조건에 기록된 이력이 없습니다.</div>
            ) : (
              historyLogs.map(log => (
                <div key={log.id} style={historyCardStyle}>
                  <div style={historyImageWrapper}>
                    {log.image_path ? (
                      <img 
                        src={`http://${host}:5000${log.image_path}`} 
                        style={historyImage} 
                        alt="event" 
                        onClick={() => window.open(`http://${host}:5000${log.image_path}`)} 
                      />
                    ) : (
                      <div style={noImgStyle}>NO IMG</div>
                    )}
                  </div>
                  <div style={historyInfo}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                        <span style={{ 
                          ...typeBadge, 
                          background: log.type === 'manual' ? '#1e3a8a' : '#450a0a', 
                          color: log.type === 'manual' ? '#60a5fa' : '#f87171' 
                        }}>
                          {log.type === 'manual' ? '수동 기록' : log.type.toUpperCase()}
                        </span>
                        {/* ✅ 시뮬레이션 배지 추가 */}
                        {log.is_simulation && (
                          <span style={{ ...typeBadge, background: '#334155', color: '#cbd5e1', fontSize: '9px' }}>TEST</span>
                        )}
                      </div>
                      <span style={resolvedByStyle}>
                        👤 {log.resolved_by ? log.resolved_by + ' 관리자' : '시스템'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                      <div style={historyAddress}>{log.address}</div>
                      <span style={{ fontSize: '11px', color: '#475569' }}>{log.time}</span>
                    </div>
                    <div style={historyMemo} title={log.memo || log.feedback_msg}>
                      "{log.memo || log.feedback_msg || "기록된 메모 없음"}"
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// --- CSS-in-JS 스타일 ---
const mainContainerStyle = { width: '100%', padding: '20px', background: '#020617', minHeight: '100%', boxSizing: 'border-box' };
const tabGroupStyle = { display: 'flex', background: '#0f172a', padding: '4px', borderRadius: '8px', border: '1px solid #1e293b' };
const baseTabBtn = { padding: '6px 16px', borderRadius: '6px', fontSize: '13px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.2s', border: 'none' };
const activeTabBtn = { ...baseTabBtn, background: '#2563eb', color: '#fff' };
const inactiveTabBtn = { ...baseTabBtn, background: 'transparent', color: '#64748b' };

// ✅ 데이터 스위칭 버튼용 스타일
const dataBtnBase = { padding: '4px 12px', borderRadius: '4px', fontSize: '12px', fontWeight: '500', cursor: 'pointer', border: '1px solid #1e293b', transition: 'all 0.2s' };
const activeDataBtn = { ...dataBtnBase, background: '#334155', color: '#fff', borderColor: '#475569' };
const inactiveDataBtn = { ...dataBtnBase, background: 'transparent', color: '#64748b' };

const filterBar = { display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' };
const datePickerStyle = { background: '#0f172a', border: '1px solid #334155', color: '#fff', padding: '5px 10px', borderRadius: '6px', outline: 'none', cursor: 'pointer' };

const logListContainer = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '15px', overflowY: 'auto' };
const historyCardStyle = { background: '#0f172a', borderRadius: '10px', border: '1px solid #1e293b', overflow: 'hidden', display: 'flex', height: '120px', transition: 'transform 0.2s' };
const historyImageWrapper = { width: '120px', height: '120px', flexShrink: 0, background: '#000', cursor: 'pointer' };
const historyImage = { width: '100%', height: '100%', objectFit: 'cover' };
const historyInfo = { flex: 1, padding: '12px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minWidth: 0 };

const typeBadge = { fontSize: '10px', padding: '2px 6px', borderRadius: '4px', fontWeight: 'bold' };
const resolvedByStyle = { fontSize: '11px', color: '#94a3b8', background: '#1e293b', padding: '2px 8px', borderRadius: '12px', fontWeight: '500', border: '1px solid #334155', whiteSpace: 'nowrap' };

const historyAddress = { fontSize: '12px', color: '#cbd5e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '150px' };
const historyMemo = { fontSize: '13px', color: '#94a3b8', fontStyle: 'italic', marginTop: '8px', borderLeft: '2px solid #334155', paddingLeft: '8px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' };

const panelWrapper = { background: '#0f172a', borderRadius: '12px', border: '1px solid #1e293b' };
const statCardStyle = { background: '#0f172a', padding: '20px', borderRadius: '12px', border: '1px solid #1e293b' };
const statTitleStyle = { color: '#64748b', fontSize: '13px', fontWeight: '600', marginBottom: '8px' };
const statValueStyle = { color: '#fff', fontSize: '26px', fontWeight: '800' };
const chartTitleStyle = { color: '#cbd5e1', fontSize: '16px', fontWeight: '700', marginBottom: '15px' };
const chartPlaceholderStyle = { height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', border: '1px dashed #334155', borderRadius: '8px' };
const labelRowStyle = { display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '8px' };
const barBackgroundStyle = { width: '100%', height: '8px', background: '#1e293b', borderRadius: '4px' };
const barProgressStyle = { height: '100%', borderRadius: '4px', transition: 'width 0.8s ease' };
const emptyStyle = { gridColumn: '1 / -1', textAlign: 'center', padding: '50px', color: '#475569' };
const noImgStyle = { height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#1e293b', fontSize: '10px' };

export default StatsPanel;