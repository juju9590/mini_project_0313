import threading

class DetectionManager:
    def __init__(self):
        self.active_detectors = {}  # { 'cctv_name': detector_instance }
        self.threads = {}           # { 'cctv_name': thread_instance }
        self._lock = threading.Lock() # 동시 접근 방지를 위한 락

    def get_or_create(self, name, detector_class, **kwargs):
        """이미 실행 중인 분석기가 있다면 반환, 없다면 생성 및 스레드 실행"""
        with self._lock:  # 여러 요청이 동시에 들어올 때 하나씩 처리
            # 1. 이미 존재하는 경우
            if name in self.active_detectors:
                # 스레드가 여전히 살아있는지 확인
                if self.threads[name].is_alive():
                    # print(f"✅ [Manager] {name} 분석기가 이미 동작 중입니다.")
                    return self.active_detectors[name]
                else:
                    print(f"⚠️ [Manager] {name} 스레드가 중단됨을 감지. 재시작합니다.")
                    # 죽은 데이터 삭제 후 아래 생성 로직으로 진행
                    del self.active_detectors[name]
                    del self.threads[name]

            # 2. 신규 생성 및 실행
            print(f"🚀 [Manager] {name} 분석기 신규 생성 및 스레드 시작")
            
            # 객체 생성 (BaseDetector 상속 구조 필수)
            instance = detector_class(name, **kwargs)
            
            # 메인 분석(run) 스레드 설정 및 시작
            t = threading.Thread(target=instance.run, name=f"Thread_{name}", daemon=True)
            t.start()
            
            self.active_detectors[name] = instance
            self.threads[name] = t
            
            return instance

    def stop_all(self):
        """서버 종료 시 모든 분석기 정지"""
        with self._lock:
            print(f"🛑 [Manager] 모든 분석기를 정지합니다. (총 {len(self.active_detectors)}개)")
            for name, instance in self.active_detectors.items():
                instance.stop()
            self.active_detectors.clear()
            self.threads.clear()

# 전역 인스턴스
detector_manager = DetectionManager()