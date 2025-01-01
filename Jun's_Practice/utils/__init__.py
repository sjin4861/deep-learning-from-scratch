# utils/__init__.py

from .functions import softmax, cross_entropy_error  # 필요한 함수만 가져오기

__all__ = ['softmax', 'cross_entropy_error']  # 외부에서 접근 가능한 함수 명시