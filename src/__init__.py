"""
MINERVA 투자 챗봇 시스템 패키지
- 사용자 성향 분석, 금융 데이터 처리, LLM 서비스 통합
"""

__version__ = '0.1.0'

# 상대 임포트만 사용 (오류 방지를 위해 try-except 사용)
try:
    from .user_profile_analyzer import UserProfileAnalyzer
except ImportError:
    pass

try:
    from .financial_data_processor import FinancialDataProcessor
except ImportError:
    pass

try:
    from .llm_service import LLMService
except ImportError:
    pass

try:
    from .memory_manager import MemoryManager
except ImportError:
    pass

try:
    from .simplified_portfolio_prediction import extract_portfolio_tickers, analyze_portfolio_with_user_profile
except ImportError:
    pass

# 새로운 국내주식 데이터 분석 모듈들
try:
    from .korean_stock_data_processor import KoreanStockDataProcessor, DataConfig
except ImportError:
    pass

try:
    from .stock_evaluator import StockEvaluator
except ImportError:
    pass

try:
    from .stock_search_engine import StockSearchEngine
except ImportError:
    pass