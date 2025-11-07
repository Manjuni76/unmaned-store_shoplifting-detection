"""
Utility functions for path setup and common operations
"""
import sys
import os


def setup_model_path():
    """
    Add model directory to sys.path for imports
    This avoids duplicating path setup code across scripts
    """
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    return model_path


def get_project_root():
    """
    Get the project root directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_ai_model_dir():
    """
    Get the ai_model directory path
    """
    return os.path.dirname(os.path.abspath(__file__))
