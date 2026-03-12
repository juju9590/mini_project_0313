/* eslint-disable */
import React from 'react';

const ControlPanel = ({ activeTab, startSim, logs }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', height: '100%', minHeight: 0 }}>
      {/* 상단 버튼부 */}
      <div style={{ background: '#111827', padding: '15px', borderRadius: '12px', border: '1px solid #1e293b', flexShrink: 0 }}>
        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#cbd5e1', marginBottom: '15px' }}>⚙️ 상황 시뮬레이션 제어</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button onClick={() => startSim("fire")} style={btnStyleFire}>🔥 화재 상황 공유</button>
          <button onClick={() => startSim("reverse")} style={btnStyleReverse}>🚗 역주행 상황 공유</button>
        </div>
      </div>

      {/* 하단 로그부: flex: 1로 영역 확장 */}
      <div style={{ 
        flex: 1, 
        background: '#111827', 
        padding: '10px', 
        borderRadius: '12px', 
        border: '1px solid #1e293b', 
        display: 'flex', 
        flexDirection: 'column', 
        minHeight: 0 
      }}>
        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#cbd5e1', marginBottom: '10px', flexShrink: 0 }}>📡 실시간 시스템 로그</div>
        <div style={{ 
          flex: 1, 
          background: '#020617', 
          padding: '12px', 
          borderRadius: '8px', 
          overflowY: 'auto', 
          minHeight: 0 
        }}>
          {logs.length === 0 ? (
            <div style={{ color: '#475569', fontSize: '11px', fontFamily: 'monospace' }}>[시스템] 모니터링 중...</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} style={{ 
                color: log.includes('🚨') ? '#f87171' : '#4ade80', 
                fontSize: '11px', 
                marginBottom: '4px', 
                fontFamily: 'monospace',
                lineHeight: '1.4'
              }}>
                {log}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const btnStyleFire = { padding: '5px', background: '#e67e22', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 'bold' };
const btnStyleReverse = { padding: '5px', background: '#f1c40f', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 'bold' };

export default ControlPanel;