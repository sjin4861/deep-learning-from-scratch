# 『밑바닥부터 시작하는 딥러닝 ❷』

<a href="http://www.yes24.com/Product/Goods/72173703"><img src="https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/cover.png" width="150" align=right></a>

[미리보기](http://preview2.hanbit.co.kr/books/zcau/) | [알려진 오류(정오표)](https://docs.google.com/document/d/1pzeh5nrP6y6A5WgT9vvxMpe-ai7ZRhU84BdAhdJzuFk/edit?usp=sharing) | [본문 그림과 수식 이미지 모음](https://github.com/WegraLee/deep-learning-from-scratch-2/raw/refs/heads/master/equations_and_figures_2.zip)

1편에서 다루지 못한 순환 신경망(RNN)을 자연어 처리와 시계열 데이터 처리에 활용하는 딥러닝 기술에 초점을 맞춰 살펴봅니다. 총 8개의 장을 하나의 이야기처럼 순서대로 읽도록 꾸몄습니다. 1편의 내용을 요약한 '신경망 복습'을 첫 장에 배치하여 신경망과 파이썬 지식을 어느 정도 갖춘 분이라면 1편을 읽지 않아도 무리 없이 따라올 수 있도록 배려했습니다.

## 파일 구성

|폴더 이름 |설명                         |
|:--        |:--                          |
|ch01       |1장에서 사용하는 소스 코드 |
|ch02       |2장에서 사용하는 소스 코드    |
|...        |...                          |
|ch08       |8장에서 사용하는 소스 코드    |
|common     |공통으로 사용하는 소스 코드  |
|dataset    |데이터셋용 소스 코드 |

6장과 7장에서 사용하는 학습된 가중치 파일은 아래 URL에서 받을 수 있습니다.

<https://www.oreilly.co.jp/pub/9784873118369/BetterRnnlm.pkl>


## 요구사항
소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.

* 파이썬 3.x
* NumPy
* Matplotlib
 
또한 선택사항으로 다음 라이브러리를 사용합니다.

* SciPy
* CuPy


## 실행방법

각 장의 디렉터리로 이동한 후 python 명령을 실행하세요(**다른 디렉터리에서는 제대로 실행되지 않을 수 있습니다!**).

```
$ cd ch01
$ python train.py

$ cd ../ch05
$ python train_custom_loop.py
```

---

## 동영상 강의
수원대학교 한경훈 교수님께서 『밑바닥부터 시작하는 딥러닝』 1, 2편을 교재로 진행하신 강의를 공개해주셨습니다. 책만으로 부족하셨던 분들께 많은 도움이 되길 바랍니다.

딥러닝 I - [강의 홈페이지](https://sites.google.com/site/kyunghoonhan/deep-learning-i)

[![시리즈 1](https://img.youtube.com/vi/8Gpa_pdHrPE/0.jpg)](https://www.youtube.com/watch?v=8Gpa_pdHrPE&list=PLBiQZMT3oSxW1RS1hn2jWBgswh0nlcgQZ)

딥러닝 II - [강의 홈페이지](https://sites.google.com/site/kyunghoonhan/deep-learning-ii)

[![시리즈 1](https://img.youtube.com/vi/5fwD1p9ymx8/0.jpg)](https://www.youtube.com/watch?v=5fwD1p9ymx8&list=PLBiQZMT3oSxXNGcmAwI7vzh2LzwcwJpxU)

딥러닝 III - [강의 홈페이지](https://sites.google.com/site/kyunghoonhan/deep-learning-iii)

[![시리즈 1](https://img.youtube.com/vi/kIobK76on3s/0.jpg)](https://www.youtube.com/watch?v=kIobK76on3s&list=PLBiQZMT3oSxV3RxoFgNcUNV4R7AlvUMDx)

---

## 팬픽 - 바닷속 딥러닝 어드벤처 (5부작)

<img src="https://github.com/WegraLee/deep-learning-from-scratch-5/blob/main/posters/%E1%84%87%E1%85%A1%E1%84%83%E1%85%A1%E1%86%BA%E1%84%89%E1%85%A9%E1%86%A8%20%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%87%E1%85%A6%E1%86%AB%E1%84%8E%E1%85%A5.png?raw=true">

"<밑바닥부터 시작하는 딥러닝>의 주인공 생선들은 딥러닝 기술로 바닷속 생태계를 어떻게 혁신하고 있을까요? 어공지능의 첨단을 이끌어가는 밑시딥 생선들과 신나는 모험을 떠나보세요."

바닷속 세계를 배경으로, 해양 생물들이 자신의 특성과 필요에 맞는 딥러닝 기술을 개발하여 문제를 해결해 나가는 모험을 그린 연작 소설입니다. 시리즈를 읽으신 분은 더 많은 재미를 느끼실 수 있도록 딥러닝 요소들을 곳곳에 삽입하였습니다.

각 편의 주인공과 주제는 다음과 같습니다.

1. **시야를 찾아서**: 쏨뱅이(쏨)가 **이미지 처리 기술**을 개발하여 주변 환경을 선명하게 파악
1. **상어공주**: 괭이상어 공주(꽹)가 **자연어 처리** 기술로 돌고래 왕자와의 사랑을 쟁취
1. **DeZero의 창조자**: 나뭇잎해룡(잎룡)이 **딥러닝 프레임워크**를 만들어 기술 보급과 협업 촉진
1. **제발, 가즈아!**: 가자미(가즈아)가 **심층 강화 학습**으로 먹이가 풍부한 새로운 바다 개척
1. **피쉬카소와 천재의 초상**: 유령실고기(피쉬카소)가 **이미지 생성 모델**로 바닷속 예술계 혁신

<a href="https://www.hanbit.co.kr/channel/series/series_detail_list.html?hcs_idx=34" target="_blank" rel="noopener noreferrer">소설 보러 가기</a>

---

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.
비상용뿐 아니라 상용으로도 자유롭게 이용하실 수 있습니다.
