# Pixie - AI 투자 어드바이저

AI 기반 개인화된 투자 자문 서비스

## 🚀 주요 기능

### 1. 투자 설문 (Survey)
- 10개 질문으로 투자 성향 분석
- 개인화된 투자 프로필 생성
- 위험 성향별 맞춤 포트폴리오 제안

### 2. 투자 학습 (Learning)
- 초보자를 위한 투자 기초 교육
- 카드뉴스 형식의 쉬운 설명
- 투자 용어 사전
- 실시간 퀴즈

### 3. AI 챗봇 (Chatbot)
- 다중 AI 에이전트 시스템
- 실시간 시장 데이터 기반 조언
- 24/7 투자 상담 서비스

### 4. 주가 예측 (Stock Prediction)
- ARIMA-X 모델 기반 예측
- 7일 후 주가 예측
- 뉴스 감정 분석 통합

### 5. 뉴스/이슈 (News)
- 실시간 금융 뉴스 수집
- AI 감정 분석
- 맞춤형 뉴스 필터링

### 6. MY 투자 (My Investment)
- 개인 포트폴리오 관리
- 관심 종목 추적
- 투자 성과 분석

## 🛠 기술 스택

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **AI/ML**: OpenAI GPT, ARIMA-X
- **Database**: SQLite / Supabase
- **Charts**: Chart.js
- **Deployment**: Vercel / GitHub Pages

## 📦 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/ll3i/pixie-.git
cd pixie-
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일 생성:
```
OPENAI_API_KEY=your_openai_api_key
FLASK_SECRET_KEY=your_secret_key_minimum_32_chars
```

### 5. 애플리케이션 실행
```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## 📱 페이지 구조

```
/                   - 메인 대시보드
/survey             - 투자 설문
/learning           - 투자 학습
/chatbot            - AI 챗봇
/stock              - 주가 예측
/news               - 뉴스/이슈
/my-investment      - MY 투자
```

## 🌐 배포

### Vercel 배포
1. Vercel CLI 설치: `npm i -g vercel`
2. 배포: `vercel`
3. 프로덕션 배포: `vercel --prod`

### GitHub Pages (정적 버전)
- https://ll3i.github.io/pixie-/

## 📄 라이선스

MIT License

## 👥 기여

기여를 환영합니다! Pull Request를 보내주세요.

## 📧 문의

문제가 있거나 제안사항이 있으면 Issues를 열어주세요.