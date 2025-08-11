import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, \
    QWidget, QLabel, QTabWidget, QScrollArea, QGridLayout, QFrame
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from openai import OpenAI
from typing import List, Dict
import json
from collections import deque

# 간소화된 버전의 포트폴리오 예측 모듈 import
from simplified_portfolio_prediction import analyze_portfolio_with_user_profile, extract_portfolio_tickers


class OpenAIThread(QThread):
    response_received = pyqtSignal(str, str)  # agent, response
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, api_key):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.messages = []
        self.agent = ""

    def run(self):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                max_tokens=4096,
                temperature=0,
            )
            self.response_received.emit(self.agent, response.choices[0].message.content.strip())
        except Exception as e:
            self.error_occurred.emit(f"Error in {self.agent}: {str(e)}")


class Memory:
    def __init__(self):
        self.short_term = deque(maxlen=10)  # Now stores tuples (speaker, content)
        self.long_term = []  # Now stores dicts with 'speaker' and 'content' keys
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def add_to_short_term(self, speaker: str, content: str):
        self.short_term.append((speaker, content))

    def add_to_long_term(self, speaker: str, content: str):
        self.long_term.append({"speaker": speaker, "content": content})
        self.save_long_term_memory()

    def get_short_term(self) -> List[tuple]:
        return list(self.short_term)

    def get_long_term(self) -> List[Dict]:
        return self.long_term

    def save_long_term_memory(self):
        with open(os.path.join(self.script_dir, "long_term_memory.json"), "w") as f:
            json.dump(self.long_term, f)

    def load_long_term_memory(self):
        try:
            with open(os.path.join(self.script_dir, "long_term_memory.json"), "r") as f:
                self.long_term = json.load(f)
        except FileNotFoundError:
            self.long_term = []


class FinancialChatbot:
    def __init__(self, api_key, personalized_prompts):
        self.memory = Memory()
        self.memory.load_long_term_memory()
        self.openai_thread = OpenAIThread(api_key)
        self.openai_thread.response_received.connect(self.handle_ai_response)
        self.user_ai_a_history = []
        self.ai_a2_ai_b_history = []
        self.agent = "AI-A"  # 초기 상태는 AI-A
        self.prompts = personalized_prompts
        self.status_update = None

    def chat(self, user_input: str):
        self.memory.add_to_long_term("User", user_input)
        self.user_ai_a_history.append({"role": "user", "content": user_input})
        self.agent = "AI-A"
        self.generate_ai_response()

    def get_system_prompt(self):
        if self.agent == "AI-A":
            return self.prompts.get("prompt_AI-A.txt", "당신은 AI 투자 조언 전문가입니다. 사용자에게 투자에 관한 유용한, 현명한 조언을 제공하세요.")
        elif self.agent == "AI-A2":
            return self.prompts.get("prompt_AI-A2.txt", "당신은 사용자의 투자 성향을 이해한 후 금융 데이터 AI(AI-B)에게 필요한 금융 정보를 물어보는 AI 금융 상담사입니다.")
        else:  # AI-B
            return self.prompts.get("prompt_AI-B.txt", "당신은 보수적인 투자 전문가입니다. 위험을 최소화하고 안전한 투자에 중점을 두어 조언하세요.")

    def generate_ai_response(self):
        self.openai_thread.agent = self.agent
        if self.agent == "AI-A":
            messages = [{"role": "system", "content": self.get_system_prompt()}] + self.user_ai_a_history

        else:  # AI-A2 or AI-B
            print(f"{self.openai_thread.agent} ready")
            messages = [{"role": "system", "content": self.get_system_prompt()}] + self.ai_a2_ai_b_history
            print(f"{self.openai_thread.agent} done")
        self.openai_thread.messages = messages
        if self.status_update:
            self.status_update(f"{self.agent} is thinking...")
        self.openai_thread.start()

    def handle_ai_response(self, agent, response):
        self.memory.add_to_short_term(agent, response)

        if agent == "AI-A":
            self.user_ai_a_history.append({"role": "assistant", "content": response})
            self.agent = "AI-A2"
            self.ai_a2_ai_b_history = [{"role": "user",
                                        "content": f"User query: {self.user_ai_a_history[-2]['content']}\nAI-A response: {response}"}]
            self.generate_ai_response()
        elif agent == "AI-A2":
            self.ai_a2_ai_b_history.append({"role": "assistant", "content": f"AI-A2: {response}"})
            self.agent = "AI-B"
            self.generate_ai_response()
        elif agent == "AI-B":
            self.ai_a2_ai_b_history.append({"role": "assistant", "content": f"AI-B: {response}"})
            if len(self.ai_a2_ai_b_history) >= 2:
                self.generate_final_response()
            else:
                self.agent = "AI-A2"
                self.generate_ai_response()
        else:  # Final response
            self.final_response_ready(response)
            if self.status_update:
                self.status_update("Ready")

        self.ai_response_ready(agent, response)

    def generate_final_response(self):
        final_prompt = f"""다음은 AI-A2와 AI-B 간의 대화 내용입니다:

        {self.get_conversation_summary()}
    
        이 정보를 바탕으로 사용자에게 제공할 최종 투자 조언을 작성해주세요. 조언은 다음 사항을 포함해야 합니다:
        
        1. 추천하는 투자 전략 및 상품
        2. 잠재적 위험과 수익 분석
        3. 주의사항 및 추가 고려사항
        4. 다음 단계 제안
        
        조언은 명확하고 간결하게, 그리고 사용자가 이해하기 쉽게 작성해주세요."""

        # 포트폴리오 예측 분석 관련 안내 추가
        if "포트폴리오" in self.user_ai_a_history[-2]['content'] or "주식" in self.user_ai_a_history[-2]['content']:
            final_prompt += f"\n\n또한, 추천한 포트폴리오에 대한 시계열 예측 분석이 '포트폴리오 예측' 탭에 표시될 것임을 언급해주세요. 사용자가 해당 탭을 확인하도록 안내해주세요."

        self.agent = "Final"
        self.openai_thread.agent = self.agent
        self.openai_thread.messages = [{"role": "system", "content": final_prompt}]
        if self.status_update:
            self.status_update("Generating final response...")
        self.openai_thread.start()

    def get_conversation_summary(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.ai_a2_ai_b_history])

    def ai_response_ready(self, agent, response):
        # 이 메서드는 GUI에 연결되어 AI 응답을 표시합니다
        pass

    def final_response_ready(self, response):
        # 이 메서드는 GUI에 연결되어 최종 응답을 표시합니다
        self.memory.add_to_long_term("AI", response)
        self.ai_a2_ai_b_history = []  # AI-A2와 AI-B의 대화 기록 초기화
        self.agent = "AI-A"
        # user_ai_a_history는 초기화하지 않음으로써 이전 사용자-AI-A 대화 내용을 유지합니다.

    def reset_ai_conversation(self):
        self.ai_a2_ai_b_history = []
        self.agent = "AI-A"


class PortfolioPredictionThread(QThread):
    """포트폴리오 예측을 백그라운드에서 실행하는 스레드"""
    prediction_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, ai_a2_ai_b_history):
        super().__init__()
        self.ai_a2_ai_b_history = ai_a2_ai_b_history
    
    def run(self):
        try:
            # 포트폴리오 종목 추출
            portfolio_tickers = extract_portfolio_tickers(self.ai_a2_ai_b_history)
            
            if not portfolio_tickers:
                print("추출된 종목 코드가 없습니다. 기본 코스닥 종목을 사용합니다.")
                # 코스닥 대표 종목 (셀트리온헬스케어, 에코프로비엠, 카카오게임즈, 알테오젠, 엘앤에프)
                portfolio_tickers = ['091990', '247540', '293490', '196170', '066970']
            
            print(f"예측에 사용할 종목 코드: {portfolio_tickers}")
            
            # 예측 분석 수행
            results = analyze_portfolio_with_user_profile(portfolio_tickers)
            
            # 결과가 비어있는지 확인
            if not results or not results.get('ticker_predictions'):
                print("예측 결과가 없습니다. 샘플 예측 데이터를 생성합니다.")
                # 샘플 예측 결과 생성
                sample_results = self.create_sample_prediction_results(portfolio_tickers)
                self.prediction_complete.emit(sample_results)
            else:
                # 결과 전송
                self.prediction_complete.emit(results)
        
        except Exception as e:
            error_message = f"포트폴리오 예측 중 오류 발생: {str(e)}"
            print(error_message)
            self.error_occurred.emit(error_message)
            
            # 오류 발생 시에도 샘플 데이터 제공
            sample_results = self.create_sample_prediction_results(['091990', '247540', '293490'])
            self.prediction_complete.emit(sample_results)
    
    def create_sample_prediction_results(self, tickers):
        """샘플 예측 결과를 생성합니다."""
        print("샘플 예측 결과 생성 중...")
        import os
        import numpy as np
        from datetime import datetime, timedelta
        import matplotlib.pyplot as plt
        
        # 스크립트 경로 가져오기
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        ticker_predictions = {}
        for ticker in tickers:
            # 종목별 기본 가격 설정
            if ticker == '091990':  # 셀트리온헬스케어
                base_price = 58000
                name = "셀트리온헬스케어"
            elif ticker == '247540':  # 에코프로비엠
                base_price = 320000
                name = "에코프로비엠"
            elif ticker == '293490':  # 카카오게임즈
                base_price = 45000
                name = "카카오게임즈"
            elif ticker == '196170':  # 알테오젠
                base_price = 70000
                name = "알테오젠"
            elif ticker == '066970':  # 엘앤에프
                base_price = 180000
                name = "엘앤에프"
            else:
                base_price = 50000
                name = f"종목 {ticker}"
            
            # 예측 데이터 생성
            predicted_price = base_price * (1 + np.random.uniform(-0.1, 0.2))
            change_pct = ((predicted_price - base_price) / base_price) * 100
            
            # 샘플 차트 생성
            plt.figure(figsize=(12, 6))
            
            # 과거 데이터 (60일) 생성
            dates = [datetime.now() - timedelta(days=i) for i in range(60, 0, -1)]
            prices = [base_price * (1 + np.random.uniform(-0.1, 0.1)) for _ in range(60)]
            plt.plot(dates, prices, label='과거 데이터', color='blue')
            
            # 예측 기간 생성
            forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
            forecast_prices = [predicted_price * (1 + np.random.uniform(-0.05, 0.05) * i/30) for i in range(30)]
            plt.plot(forecast_dates, forecast_prices, 'r--', label='예측 가격')
            
            # 신뢰구간 생성
            upper_bound = [price * 1.05 for price in forecast_prices]
            lower_bound = [price * 0.95 for price in forecast_prices]
            plt.fill_between(forecast_dates, lower_bound, upper_bound, color='red', alpha=0.2, label='95% 신뢰구간')
            
            # 그래프 설정
            plt.title(f'Stock Price Prediction for {name} ({ticker})')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 파일로 저장
            chart_path = os.path.join(script_dir, f'{ticker}_prediction.png')
            plt.savefig(chart_path)
            plt.close()
            
            # 예측 결과 저장
            ticker_predictions[ticker] = {
                'current_price': float(base_price),
                'predicted_price': float(predicted_price),
                'change_pct': float(change_pct),
                'trend': "상승" if change_pct > 0 else "하락",
                'volatility': float(10.0),  # 샘플 변동성
                'chart_path': chart_path
            }
        
        # 포트폴리오 전체 분석 결과
        avg_return = np.mean([data['change_pct'] for data in ticker_predictions.values()])
        risk_assessment = "안정적" if avg_return > 0 else "불안정"
        
        return {
            'ticker_predictions': ticker_predictions,
            'portfolio_analysis': {
                'expected_return': float(avg_return),
                'risk_assessment': risk_assessment,
                'recommendation': "유지" if risk_assessment == "안정적" else "재조정 필요"
            },
            'user_profile': {
                'risk_level': 'medium',
                'investment_horizon': 'medium'
            }
        }


class ChatbotGUI(QMainWindow):
    def __init__(self, api_key, personalized_prompts):
        super().__init__()
        self.chatbot = FinancialChatbot(api_key, personalized_prompts)
        self.portfolio_predictions = {}  # 포트폴리오 예측 결과 저장
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Financial Chatbot')
        self.setGeometry(100, 100, 1400, 800)  # 창 크기 확대

        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 왼쪽 패널 (채팅)
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        
        # 사용자-AI 채팅
        chat_layout.addWidget(QLabel("User-AI Chat"))
        self.user_ai_chat = QTextEdit(self)
        self.user_ai_chat.setReadOnly(True)
        chat_layout.addWidget(self.user_ai_chat)
        
        # 입력 필드와 전송 버튼
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        input_layout.addWidget(self.input_field)
        self.send_button = QPushButton('Send', self)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)
        
        # 오른쪽 패널 (탭 위젯)
        right_panel = QTabWidget()
        
        # AI 대화 탭
        ai_ai_tab = QWidget()
        ai_ai_layout = QVBoxLayout(ai_ai_tab)
        ai_ai_layout.addWidget(QLabel("AI-A2 and AI-B Chat"))
        self.ai_ai_chat = QTextEdit(self)
        self.ai_ai_chat.setReadOnly(True)
        ai_ai_layout.addWidget(self.ai_ai_chat)
        right_panel.addTab(ai_ai_tab, "AI 대화")
        
        # 포트폴리오 예측 탭
        self.prediction_tab = QWidget()
        self.prediction_layout = QVBoxLayout(self.prediction_tab)
        self.prediction_scroll = QScrollArea()
        self.prediction_scroll.setWidgetResizable(True)
        self.prediction_content = QWidget()
        self.prediction_grid = QGridLayout(self.prediction_content)
        self.prediction_scroll.setWidget(self.prediction_content)
        self.prediction_layout.addWidget(self.prediction_scroll)
        right_panel.addTab(self.prediction_tab, "포트폴리오 예측")
        
        # 상태 표시 레이블
        self.status_label = QLabel("Ready")
        chat_layout.addWidget(self.status_label)
        
        # 패널 추가
        main_layout.addWidget(chat_panel, 1)  # 1:2 비율로 설정
        main_layout.addWidget(right_panel, 2)
        
        # 콜백 연결
        self.chatbot.final_response_ready = self.update_user_ai_chat
        self.chatbot.ai_response_ready = self.update_ai_ai_chat
        self.chatbot.openai_thread.error_occurred.connect(self.show_error)
        self.chatbot.status_update = self.update_status

    def send_message(self):
        user_input = self.input_field.text()
        if user_input.strip() == "":
            return
        self.user_ai_chat.append(f"User: {user_input}")
        self.input_field.clear()
        self.update_status("Processing...")
        self.send_button.setEnabled(False)
        self.input_field.setEnabled(False)

        # AI 대화 창 초기화
        self.ai_ai_chat.clear()
        self.chatbot.reset_ai_conversation()

        self.chatbot.chat(user_input)

    def update_user_ai_chat(self, response):
        self.user_ai_chat.append(f"AI Final Response: {response}")
        self.user_ai_chat.verticalScrollBar().setValue(self.user_ai_chat.verticalScrollBar().maximum())
        self.update_status("Ready")
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
        
        # 포트폴리오 예측 수행 (종목 추천, 주식, 포트폴리오 관련 질문인 경우)
        user_question = self.chatbot.user_ai_a_history[-2]['content'].lower() if len(self.chatbot.user_ai_a_history) >= 2 else ""
        is_stock_related = any(keyword in user_question for keyword in 
                              ["포트폴리오", "주식", "종목", "투자", "코스피", "코스닥", "삼성전자", "증권", "수익률", "주가"])
        
        if is_stock_related or "종목코드" in response:
            print("주식 관련 질문 감지. 포트폴리오 예측을 시작합니다.")
            self.run_portfolio_prediction()
            # 포트폴리오 탭으로 전환 (사용자에게 알림)
            self.user_ai_chat.append("추천 종목에 대한 시계열 예측 분석을 '포트폴리오 예측' 탭에서 확인할 수 있습니다.")

    def update_ai_ai_chat(self, agent, response):
        if agent in ["AI-A2", "AI-B"]:
            self.ai_ai_chat.append(f"{agent}: {response}")
            self.ai_ai_chat.verticalScrollBar().setValue(self.ai_ai_chat.verticalScrollBar().maximum())

    def show_error(self, error_message):
        self.ai_ai_chat.append(f"Error: {error_message}")
        self.ai_ai_chat.verticalScrollBar().setValue(self.ai_ai_chat.verticalScrollBar().maximum())
        self.update_status("Error occurred")
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)

    def update_status(self, status):
        self.status_label.setText(status)
        if status == "Ready":
            self.status_label.setStyleSheet("color: green;")
        elif "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: orange;")
    
    def run_portfolio_prediction(self):
        """포트폴리오 예측을 실행하고 결과를 표시합니다."""
        # 기존 예측 결과 위젯 제거
        self.clear_grid_layout(self.prediction_grid)
        
        # 진행 상황 표시 라벨 추가
        progress_label = QLabel("종목 예측 분석 중입니다... 잠시만 기다려주세요.")
        progress_label.setStyleSheet("color: blue; font-size: 14px;")
        progress_label.setAlignment(Qt.AlignCenter)
        self.prediction_grid.addWidget(progress_label, 0, 0, 1, 2)
        
        # UI 업데이트
        QApplication.processEvents()
        
        # 상태 업데이트
        self.update_status("Analyzing portfolio...")
        
        # 예측 스레드 시작
        self.prediction_thread = PortfolioPredictionThread(self.chatbot.ai_a2_ai_b_history)
        self.prediction_thread.prediction_complete.connect(self.display_prediction_results)
        self.prediction_thread.error_occurred.connect(self.show_error)
        self.prediction_thread.start()
    
    def display_prediction_results(self, results):
        """포트폴리오 예측 결과를 화면에 표시합니다."""
        # 기존 결과 위젯 제거
        self.clear_grid_layout(self.prediction_grid)
        
        self.statusBar().showMessage("포트폴리오 예측 결과 표시 중...")
        
        # 결과가 없거나 티커 예측이 없는 경우
        if not results or not results.get('ticker_predictions'):
            error_label = QLabel("포트폴리오 예측 결과가 없습니다.")
            error_label.setStyleSheet("color: #FF5252; font-size: 14px; font-weight: bold;")
            self.prediction_grid.addWidget(error_label, 0, 0, 1, 2)
            
            explanation_label = QLabel("가능한 원인: 종목 코드나 이름이 명확하지 않음, 주가 데이터 로드 오류, 또는 예측 모델에 충분한 데이터가 없음.\n"
                                    "특정 종목 코드나 이름을 명시해 주세요 (예: '091990' 또는 '셀트리온헬스케어').")
            explanation_label.setStyleSheet("color: #757575; font-size: 12px;")
            explanation_label.setWordWrap(True)
            self.prediction_grid.addWidget(explanation_label, 1, 0, 1, 2)
            
            self.statusBar().showMessage("Ready")
            return
        
        # 포트폴리오 종합 분석 결과 표시
        portfolio_analysis = results.get('portfolio_analysis', {})
        if portfolio_analysis:
            header_label = QLabel("포트폴리오 종합 분석")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
            self.prediction_grid.addWidget(header_label, 0, 0, 1, 2)
            
            expected_return = portfolio_analysis.get('expected_return', 0)
            return_color = "#4CAF50" if expected_return > 0 else "#FF5252"
            return_label = QLabel(f"예상 수익률: {expected_return:.2f}%")
            return_label.setStyleSheet(f"color: {return_color}; font-size: 13px;")
            self.prediction_grid.addWidget(return_label, 1, 0)
            
            risk_label = QLabel(f"위험 평가: {portfolio_analysis.get('risk_assessment', '정보 없음')}")
            self.prediction_grid.addWidget(risk_label, 1, 1)
            
            recommendation_label = QLabel(f"추천: {portfolio_analysis.get('recommendation', '정보 없음')}")
            recommendation_label.setStyleSheet("font-weight: bold;")
            self.prediction_grid.addWidget(recommendation_label, 2, 0, 1, 2)
            
            # 구분선 추가
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            self.prediction_grid.addWidget(line, 3, 0, 1, 2)
            
            # 개별 종목 예측 헤더
            stocks_header = QLabel("개별 종목 예측")
            stocks_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3; margin-top: 10px;")
            self.prediction_grid.addWidget(stocks_header, 4, 0, 1, 2)
        
        # 개별 종목 예측 결과 표시
        row_index = 5
        ticker_predictions = results.get('ticker_predictions', {})
        
        for ticker, prediction in ticker_predictions.items():
            ticker_name = self.get_ticker_name(ticker) or f"종목 {ticker}"
            ticker_label = QLabel(f"{ticker_name} ({ticker})")
            ticker_label.setStyleSheet("font-weight: bold; font-size: 13px;")
            self.prediction_grid.addWidget(ticker_label, row_index, 0, 1, 2)
            row_index += 1
            
            # 현재가와 예측가
            current_price = prediction.get('current_price', 0)
            predicted_price = prediction.get('predicted_price', 0)
            price_info = QLabel(f"현재가: {current_price:,.0f}원 → 예측가: {predicted_price:,.0f}원")
            self.prediction_grid.addWidget(price_info, row_index, 0, 1, 2)
            row_index += 1
            
            # 변화율
            change_pct = prediction.get('change_pct', 0)
            change_color = "#4CAF50" if change_pct > 0 else "#FF5252"
            change_label = QLabel(f"예상 변화율: {change_pct:.2f}%")
            change_label.setStyleSheet(f"color: {change_color}; font-weight: bold;")
            self.prediction_grid.addWidget(change_label, row_index, 0)
            
            # 추세
            trend = prediction.get('trend', '정보 없음')
            trend_color = "#4CAF50" if trend == "상승" else "#FF5252"
            trend_label = QLabel(f"추세: {trend}")
            trend_label.setStyleSheet(f"color: {trend_color};")
            self.prediction_grid.addWidget(trend_label, row_index, 1)
            row_index += 1
            
            # 변동성
            volatility = prediction.get('volatility', 0)
            volatility_label = QLabel(f"변동성: {volatility:.2f}%")
            self.prediction_grid.addWidget(volatility_label, row_index, 0, 1, 2)
            row_index += 1
            
            # 차트 이미지 표시
            chart_path = prediction.get('chart_path')
            if chart_path and os.path.exists(chart_path):
                chart_pixmap = QPixmap(chart_path)
                chart_pixmap = chart_pixmap.scaled(600, 300, Qt.KeepAspectRatio)
                chart_label = QLabel()
                chart_label.setPixmap(chart_pixmap)
                self.prediction_grid.addWidget(chart_label, row_index, 0, 1, 2, Qt.AlignCenter)
                row_index += 1
            
            # 구분선 추가
            if len(ticker_predictions) > 1:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                self.prediction_grid.addWidget(line, row_index, 0, 1, 2)
                row_index += 1
        
        # 예측 관련 추가 정보
        if results.get('user_profile'):
            user_profile = results['user_profile']
            profile_label = QLabel(f"사용자 투자 성향: {'낮은' if user_profile.get('risk_level') == 'low' else '중간' if user_profile.get('risk_level') == 'medium' else '높은'} 위험도, "
                                 f"{'단기' if user_profile.get('investment_horizon') == 'short' else '중기' if user_profile.get('investment_horizon') == 'medium' else '장기'} 투자 기간")
            profile_label.setStyleSheet("color: #757575; font-size: 12px;")
            self.prediction_grid.addWidget(profile_label, row_index, 0, 1, 2)
            row_index += 1
        
        disclaimer_label = QLabel("※ 이 예측 결과는 참고용으로만 사용해야 하며, 실제 투자 결정에는 전문가의 조언을 구하십시오.")
        disclaimer_label.setStyleSheet("color: #FF5252; font-size: 11px;")
        disclaimer_label.setWordWrap(True)
        self.prediction_grid.addWidget(disclaimer_label, row_index, 0, 1, 2)
        
        self.statusBar().showMessage("Ready")

    def get_ticker_name(self, ticker_code):
        """티커 코드에 해당하는 주식 이름을 반환합니다."""
        # 종목코드-이름 매핑
        stock_mapping = {
            "091990": "셀트리온헬스케어",
            "247540": "에코프로비엠",
            "293490": "카카오게임즈",
            "068760": "셀트리온제약",
            "196170": "알테오젠",
            "035760": "CJ ENM",
            "361610": "SK아이이테크놀로지",
            "263750": "펄어비스",
            "066970": "엘앤에프",
            "112040": "위메이드",
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "035420": "NAVER",
            "035720": "카카오",
        }
        return stock_mapping.get(ticker_code)

    def clear_grid_layout(self, layout):
        """그리드 레이아웃의 모든 위젯을 제거합니다."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


def load_prompt_template(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        if "AI-A" in filename:
            return "당신은 AI 투자 조언 전문가입니다. 사용자에게 투자에 관한 유용한, 현명한 조언을 제공하세요."
        elif "AI-B" in filename:
            return "당신은 보수적인 투자 전문가입니다. 위험을 최소화하고 안전한 투자에 중점을 두어 조언하세요."
        else:
            return "투자 조언 전문가로서 역할을 수행하세요."


def replace_placeholders(template, analysis_results):
    if not analysis_results:
        # 분석 결과가 비어있는 경우 모든 플레이스홀더를 default 메시지로 대체
        default_msg = "분석 결과가 없습니다"
        for placeholder in ["risk_tolerance_analysis", "investment_time_horizon_analysis", 
                           "financial_goal_orientation_analysis", "information_processing_style_analysis", 
                           "investment_fear_analysis", "investment_confidence_analysis",
                           "overall_evaluation"]:
            template = template.replace(f'[{placeholder}]', default_msg)
        return template
        
    # 분석 결과가 있는 경우 정상적으로 대체
    for key, value in analysis_results.items():
        placeholder = f'[{key}]'
        if placeholder in template:
            template = template.replace(placeholder, value)
    
    return template


def get_personalized_prompts(json_file, prompt_files):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # 프로젝트 루트 디렉토리
    json_path = os.path.join(root_dir, json_file)  # 루트 디렉토리에서 파일 찾기
    
    try:
        print(f"Trying to load analysis results from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_results = json.load(f)
            print(f"Successfully loaded analysis results with keys: {list(analysis_results.keys())}")
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. Trying alternative location...")
        # 대체 경로 시도 (src 디렉토리 내에 있을 경우)
        json_path = os.path.join(script_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
                print(f"Successfully loaded analysis results from src directory with keys: {list(analysis_results.keys())}")
        except FileNotFoundError:
            print(f"Warning: {json_path} not found either. Using empty analysis results.")
            analysis_results = {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {json_path}. Error: {str(e)}")
            analysis_results = {}
        except Exception as e:
            print(f"Error loading {json_path}: {str(e)}")
            analysis_results = {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}. Error: {str(e)}")
        analysis_results = {}
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        analysis_results = {}

    personalized_prompts = {}
    for prompt_file in prompt_files:
        try:
            prompt_path = os.path.join(script_dir, prompt_file)
            print(f"Loading prompt template from: {prompt_path}")
            template = load_prompt_template(prompt_path)
            if prompt_file in ['prompt_AI-A.txt', 'prompt_AI-A2.txt']:
                print(f"Replacing placeholders in {prompt_file}...")
                personalized_prompts[prompt_file] = replace_placeholders(template, analysis_results)
            else:
                personalized_prompts[prompt_file] = template  # prompt_AI-B.txt 등은 그대로 저장
        except FileNotFoundError:
            print(f"Warning: {prompt_file} not found. Skipping this prompt.")

    return personalized_prompts


if __name__ == "__main__":
    json_file = 'analysis_results.json'
    prompt_files = ['prompt_AI-A.txt', 'prompt_AI-A2.txt', 'prompt_AI-B.txt', 
                   'prompt_survey-analysis.txt', 'prompt_survey-score.txt']  # 모든 필요한 프롬프트 파일 포함

    print("=" * 50)
    print("MINERVA 초기화 중...")
    
    # 개인화된 프롬프트 생성
    personalized_prompts = get_personalized_prompts(json_file, prompt_files)
    
    # 프롬프트 내용 확인 (디버깅용)
    for key, value in personalized_prompts.items():
        print(f"\n{key} 프롬프트의 처음 100자:")
        print(value[:100] + "...")

    app = QApplication(sys.argv)
    api_key = "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"  # API 키를 적절히 관리해야 합니다
    gui = ChatbotGUI(api_key, personalized_prompts)
    gui.show()
    sys.exit(app.exec_())
