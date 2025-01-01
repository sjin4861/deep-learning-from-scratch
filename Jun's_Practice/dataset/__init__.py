# MNIST 데이터셋을 불러오는 함수를 제공하는 패키지

from .mnist import load_mnist  # 별도의 파일로 함수 분리

__all__ = ['load_mnist']