import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  envDir: '../',
  server: {
    // [핵심] 모든 네트워크 인터페이스를 허용하여 외부 접속을 가능하게 합니다.
    host: '0.0.0.0', 
    port: 5173,
    // 필요 시 CORS 에러 방지를 위한 설정
    cors: true,
    allowedHosts: [
      '.ts.net' // 모든 Tailscale 주소를 허용하고 싶을 때 (마침표 포함)
    ]
  }
})