"""
Pixie - AI 투자 어드바이저 (Vercel 배포용 경량 버전)
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv
import random

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-min-32-chars-long-for-security')

# 세션 설정
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24시간

# 샘플 데이터 (실제 DB 대신 사용)
SAMPLE_STOCKS = [
    {"ticker": "005930", "name": "삼성전자", "price": 72500, "change": -1.2},
    {"ticker": "000660", "name": "SK하이닉스", "price": 124500, "change": 2.3},
    {"ticker": "035720", "name": "카카오", "price": 45200, "change": -0.5},
    {"ticker": "035420", "name": "NAVER", "price": 215000, "change": 1.8},
    {"ticker": "051910", "name": "LG화학", "price": 542000, "change": 0.7}
]

SAMPLE_NEWS = [
    {
        "title": "삼성전자, AI 반도체 투자 확대",
        "summary": "삼성전자가 차세대 AI 반도체 생산을 위해 대규모 투자를 발표했습니다.",
        "date": datetime.now().isoformat(),
        "sentiment": "positive"
    },
    {
        "title": "미국 금리 인상 우려 지속",
        "summary": "연준의 추가 금리 인상 가능성으로 시장 변동성이 확대되고 있습니다.",
        "date": datetime.now().isoformat(),
        "sentiment": "negative"
    }
]

# 라우트 정의
@app.route('/')
def index():
    """메인 페이지"""
    # Vercel에서는 정적 HTML 직접 반환
    import os
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return render_template('index.html')

@app.route('/survey')
def survey():
    """투자 설문 페이지"""
    return render_template('survey.html')

@app.route('/learning')
def learning():
    """투자 학습 페이지"""
    return render_template('learning.html')

@app.route('/chatbot')
def chatbot():
    """AI 챗봇 페이지"""
    return render_template('chatbot.html')

@app.route('/stock')
def stock():
    """주가 예측 페이지"""
    return render_template('stock.html')

@app.route('/news')
def news():
    """뉴스/이슈 페이지"""
    return render_template('news.html')

@app.route('/my-investment')
def my_investment():
    """MY 투자 페이지"""
    return render_template('my_invest.html')

# API 엔드포인트
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """챗봇 API (간단한 응답)"""
    data = request.get_json()
    user_message = data.get('message', '')
    
    # 간단한 응답 생성
    response = {
        'success': True,
        'message': f"투자 관련 질문 '{user_message}'에 대한 답변을 준비 중입니다. Vercel 데모 버전에서는 제한된 기능만 제공됩니다.",
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(response)

@app.route('/api/stocks')
def api_stocks():
    """주식 목록 API"""
    return jsonify({
        'success': True,
        'stocks': SAMPLE_STOCKS
    })

@app.route('/api/news')
def api_news():
    """뉴스 목록 API"""
    return jsonify({
        'success': True,
        'news': SAMPLE_NEWS
    })

@app.route('/api/predictions')
def api_predictions():
    """주가 예측 API (샘플 데이터)"""
    predictions = []
    for stock in SAMPLE_STOCKS[:3]:
        predictions.append({
            'stock_code': stock['ticker'],
            'stock_name': stock['name'],
            'current_price': stock['price'],
            'predicted_price': int(stock['price'] * (1 + stock['change']/100)),
            'change_percent': stock['change'],
            'confidence': 'medium',
            'trend': 'bullish' if stock['change'] > 0 else 'bearish'
        })
    
    return jsonify({
        'success': True,
        'time_series_forecasts': predictions
    })

@app.route('/api/user-profile')
def api_user_profile():
    """사용자 프로필 API"""
    # 세션에서 프로필 가져오기 또는 샘플 데이터
    profile = session.get('user_profile', {
        'type_label': '균형형 투자자',
        'risk_score': 50,
        'portfolio_type': '균형형'
    })
    
    return jsonify({
        'success': True,
        'profile': profile
    })

@app.route('/api/survey/submit', methods=['POST'])
def api_survey_submit():
    """설문 제출 API"""
    data = request.get_json()
    answers = data.get('answers', {})
    
    # 간단한 점수 계산
    risk_score = sum([int(v) for v in answers.values() if v.isdigit()]) / len(answers) * 20
    
    # 투자 성향 결정
    if risk_score >= 70:
        type_label = '공격적 투자자'
    elif risk_score >= 50:
        type_label = '적극적 투자자'
    elif risk_score >= 30:
        type_label = '균형형 투자자'
    else:
        type_label = '안정형 투자자'
    
    profile = {
        'type_label': type_label,
        'risk_score': risk_score,
        'portfolio_type': type_label.replace(' 투자자', '')
    }
    
    # 세션에 저장
    session['user_profile'] = profile
    
    return jsonify({
        'success': True,
        'profile': profile,
        'message': '투자 성향 분석이 완료되었습니다.'
    })

@app.route('/api/watchlist')
def api_watchlist():
    """관심 종목 API"""
    watchlist = session.get('watchlist', SAMPLE_STOCKS[:2])
    return jsonify({
        'success': True,
        'watchlist': watchlist
    })

@app.route('/api/watchlist', methods=['POST'])
def api_watchlist_add():
    """관심 종목 추가"""
    data = request.get_json()
    ticker = data.get('ticker')
    name = data.get('name')
    
    watchlist = session.get('watchlist', [])
    watchlist.append({'ticker': ticker, 'name': name, 'price': 0, 'change_percent': 0})
    session['watchlist'] = watchlist
    
    return jsonify({'success': True, 'message': '관심 종목에 추가되었습니다.'})

@app.route('/api/learning/terms')
def api_learning_terms():
    """투자 용어 API"""
    terms = [
        {
            'id': 1,
            'term': 'PER',
            'description': '주가수익비율. 주가를 주당순이익으로 나눈 값',
            'category': '기본'
        },
        {
            'id': 2,
            'term': 'PBR',
            'description': '주가순자산비율. 주가를 주당순자산으로 나눈 값',
            'category': '기본'
        }
    ]
    return jsonify({'success': True, 'terms': terms})

@app.route('/api/learning/quiz')
def api_learning_quiz():
    """퀴즈 API"""
    quiz = {
        'question': 'PER이 낮을수록 어떤 의미일까요?',
        'options': [
            '주가가 저평가되어 있다',
            '주가가 고평가되어 있다',
            '회사가 적자다',
            '배당금이 많다'
        ],
        'correct': 0
    }
    return jsonify({'success': True, 'quiz': quiz})

# 건강 체크
@app.route('/health')
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Vercel을 위한 앱 export
app = app

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)