import logging
import sys
from typing import Optional


class Logger:
    """Classes responsible for setting up and managing logging (SRP)"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, level: str = "INFO") -> logging.Logger:
        """Return Logger Instances (Singletone Pattern)"""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name, level)
        return cls._loggers[name]
    
    @classmethod
    def _create_logger(cls, name: str, level: str) -> logging.Logger:
        """Create a new logger"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplication if the handler already exists
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @classmethod
    def set_level(cls, name: str, level: str):
        """Changing the Logger Level"""
        if name in cls._loggers:
            cls._loggers[name].setLevel(getattr(logging, level.upper()))