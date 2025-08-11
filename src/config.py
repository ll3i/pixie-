#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Centralized configuration management for Investment Chatbot.
- Environment variable management with validation
- Security configuration enforcement
- API key management with fallback modes
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Secure configuration management following cursor rules."""
    
    def __init__(self):
        load_dotenv()
        self._config = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self) -> None:
        """Load configuration from environment variables."""
        # REQUIRED: 안전한 .env 파일 로드
        try:
            load_dotenv()
        except Exception as e:
            logger.warning(f".env 파일 로드 실패, 환경변수만 사용: {e}")
        
        self._config = {
            # REQUIRED: API Keys
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
            'CLOVA_API_KEY': os.environ.get('CLOVA_API_KEY', ''),
            
            # REQUIRED: Database Configuration
            'SUPABASE_URL': os.environ.get('SUPABASE_URL', ''),
            'SUPABASE_KEY': os.environ.get('SUPABASE_KEY', ''),
            
            # REQUIRED: Application Security
            'FLASK_SECRET_KEY': os.environ.get('FLASK_SECRET_KEY', ''),
            'ENCRYPTION_KEY': os.environ.get('ENCRYPTION_KEY', ''),
            
            # REQUIRED: Rate Limiting
            'MAX_REQUESTS_PER_MINUTE': int(os.environ.get('MAX_REQUESTS_PER_MINUTE', '60')),
            'MAX_RETRIES': int(os.environ.get('MAX_RETRIES', '3')),
            'RETRY_DELAY': int(os.environ.get('RETRY_DELAY', '1')),
            
            # REQUIRED: Application Settings
            'FLASK_ENV': os.environ.get('FLASK_ENV', 'development'),
            'DEBUG': os.environ.get('DEBUG', 'false').lower() == 'true',
            'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
        }
    
    def _validate_config(self) -> None:
        """Validate required configuration is present."""
        # REQUIRED: Basic validation
        required_keys = ['FLASK_SECRET_KEY']
        
        # REQUIRED: Production validation
        if self._config['FLASK_ENV'] == 'production':
            required_keys.extend(['SUPABASE_URL', 'SUPABASE_KEY'])
        
        missing_keys = [key for key in required_keys if not self._config.get(key)]
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {missing_keys}")
        
        # REQUIRED: Validate key formats
        if self._config['SUPABASE_URL'] and not self._config['SUPABASE_URL'].startswith('https://'):
            raise ValueError("SUPABASE_URL must use HTTPS")
        
        if len(self._config['FLASK_SECRET_KEY']) < 32:
            raise ValueError("FLASK_SECRET_KEY must be at least 32 characters")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value safely."""
        return self._config.get(key, default)
    
    def is_api_available(self, api_name: str) -> bool:
        """Check if API credentials are available."""
        key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'clova': 'CLOVA_API_KEY'
        }
        
        api_key = key_mapping.get(api_name.lower())
        return bool(self._config.get(api_key))
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration."""
        return {
            'supabase_url': self._config.get('SUPABASE_URL', ''),
            'supabase_key': self._config.get('SUPABASE_KEY', ''),
            'sqlite_path': os.path.join(os.path.dirname(__file__), '..', 'newsbot.db')
        }

# REQUIRED: Global configuration instance
config = SecurityConfig()

# REQUIRED: Environment-specific configuration
class EnvironmentConfig:
    """Environment-specific configuration management."""
    
    class Development:
        DEBUG = True
        TESTING = False
        LOG_LEVEL = "DEBUG"
        RATE_LIMIT_ENABLED = False
    
    class Testing:
        DEBUG = False
        TESTING = True
        LOG_LEVEL = "WARNING"
        RATE_LIMIT_ENABLED = False
    
    class Production:
        DEBUG = False
        TESTING = False
        LOG_LEVEL = "ERROR"
        RATE_LIMIT_ENABLED = True
        
        @classmethod
        def validate(cls) -> None:
            """Validate production configuration."""
            required_vars = [
                'SUPABASE_URL', 'SUPABASE_KEY', 
                'FLASK_SECRET_KEY'
            ]
            missing = [var for var in required_vars if not os.environ.get(var)]
            if missing:
                raise ValueError(f"Missing production variables: {missing}")

def get_config() -> type:
    """Get configuration based on environment."""
    env = config.get('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        EnvironmentConfig.Production.validate()
        return EnvironmentConfig.Production
    elif env == 'testing':
        return EnvironmentConfig.Testing
    else:
        return EnvironmentConfig.Development 