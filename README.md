[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nRcUn8vA)

# 1. 프로젝트 배경
> 최근 프로축구 및 아마추어 축구 모두에서 전술 분석은 매우 중요한 요소가 되고 있습니다. 과거에는 코치와 전술 분석관이 수작업으로 축구 영상을 돌려보며 전술을 파악했지만, 이는 시간과 인력이 많이 소모되는 방식이었습니다. 인공지능과 컴퓨터 비전 기술이 발전하면서, 데이터 기반 스포츠 분석의 필요성이 높아졌고, 이로 인해 영상 기반 자동 분석이 주목받고 있습니다. 
>
> 딥러닝 기반 객체와 추적 알고리즘이 고도화되고 있습니다. 예시로 딥러닝 기반 객체 탐지에는 YOLO, Faster R-CNN등이 있고, 추적 알고리즘에는 ByteTrack, SORT, DeepSORT등이 있습니다. 이처럼 컴퓨터 비전과 AI 기술의 발전으로 경기장에서 선수 위치, 움직임, 패스 경로 등을 자동으로 추출할 수 있게 되었습니다. 
>
> 이를 통해 경기 전체의 전술적 패턴을 수치화하여 분석하는 시도과 활발히 진행되고 있습니다. 때문에 스포츠 산업 내 AI 기술 활용이 확대되고 있습니다. 프로 구단뿐만 아니라 대학팀, 아마추어 리그에서도 경기력 향상을 위해 AI를 이용한 영상 분석을 필요로 하고 있습니다.

## 1.1. 국내외 시장 현황 및 문제점
> AI의 발달로 현재 국내외로 프로축구 혹은 아마추어 축구 시장에서 이미 AI를 활용하여 축구 전술 분석이 활발하게 이루어지고 있습니다. 국내에서 제일 잘 알려져 있는 것을 BEPRO data입니다. K리그 1, 2부에서 사용하고 있습니다. 주로 선수와 경기들 데이터를 기반으로 분석하고 있습니다. 국외에서는 Tactic AI, Wyscout, Catapult, Pixellot, PlaymakerAI 등 다양한 축구 분석 솔루션이 존재합니다. 그 중에서 Tactic AI이 주로 사용되고 있습니다. 이 AI 모델은 DeepMind와 프로축구단 Liverpool FC와의 협력으로 만들어졌습니다. Tactic AI는 선수 추적 데이터 기반으로 선수 위치, 움직임, 패스 연결망 등을 분석하고 전술 패턴을 인식합니다. 실제 코치들 평가에서도 인간 전문가들보다 더 유용한 전술 분석을 조언하는 것으로 인정되었다.
> 기존 축구 경기 분석은 팀 코치의 실력에 따라 코칭의 질이 달라지는 정보의 불평등 문제가 있습니다.
> 코치가 없는 소규모 동아리나 팀 등은 전술 분석과 코칭을 받기 어렵다는 문제가 있습니다.
> 또한 기존에는 코칭에 필요한 데이터 준비와 전술 분석에 대해 오랜 시간이 걸린다는 문제가 있습니다


## 1.2. 필요성과 기대효과
> 이 연구는 축구 영상을 AI 기반으로 자동 분석을 하여 선수들의 위치 데이터와 어떠한 전술인지 파악하는 것으로 효율적이고 신속하게 축구 전술 분석을 할 수 있습니다. 코칭 스태프가 없는 일반 아마추어, 중/고등 축구에서 활용적으로 사용할 수 있습니다. 일반 코치들이 분석하는 것과 달리 빠른 전술 분석이 가능하여 전술 분석 시간이 줄어드는 기대효과가 있습니다. 

# 2. 개발 목표
## 2.1. 목표 및 세부 내용
>이번 연구의 목표는 축구 경기 영상으로 선수 트래킹을 중심으로 전술 분석을 할 수 있도록 하고, 각 선수별로 선수의 팀 소속, 팀 간 간격 유지 여부, 포지션별 밀집도와 같이 선수의 포지션 및 필드 라인 기반 분석을 하려고 합니다. 딥러닝 기반 객체 추적 및 분석 기법을 활용하여, 영상 속에서 선수의 위치와 팀, 전술 등을 식별하는 것을 목표로 하고 있습니다.
>
>본 프로젝트는 웹 기반 축구 영상 전술 분석 시스템을 개발하는 것을 목표로 합니다. 사용자가 웹 인터페이스를 통해 축구 경기 영상을 업로드하면, 서버에서 딥러닝 비전 모델을 이용해 영상 속 선수들을 자동으로 탐지 및 추적합니다. 분석이 완료된 영상은 웹페이지에서 즉시 확인하고, 결과 파일을 직접 다운로드할 수 있는 기능을 제공합니다.
>



## 2.2. 기존 서비스 대비 차별성 
> 현재 글로벌 시장에서는 Tactic AI, Wyscout, Catapult, Pixellot, PlaymakerAI 등 다양한 축구 분석 솔루션이 존재합니다. 이들 대부분은 고가의 장비(트래킹 센서, 고해상도 카메라 등)나 구독 기반 서비스에 의존하기 때문에, 프로 구단이나 재정적으로 여유 있는 팀 위주로 사용되는 경향이 있습니다. 국내에서도 비프로(Bepro)와 같은 데이터 분석 플랫폼이 존재하지만, 아직까지는 K리그 일부 구단이나 프로 레벨에 한정된 서비스에 머무르고 있습니다.  

본 프로젝트의 차별성은 다음과 같이 요약할 수 있습니다:
- **웹 기반 접근성** – 단순히 웹 브라우저와 인터넷만 있으면 별도의 장비나 고비용 라이선스 없이도 누구나 쉽게 분석을 수행할 수 있습니다.  
- **경량화된 AI 모델** – YOLOv8n, MobileNet, LSTM을 적절히 결합하여 높은 정확도를 유지하면서도 처리 속도와 비용을 최적화했습니다.  
- **분석 속도 개선** – ByteTrack을 통한 추적 기법을 활용하여 실시간에 가까운 전술 분석 피드백을 제공합니다. 

따라서 본 프로젝트는 기존 서비스 대비 저비용과 사용자 편의성이라는 측면에서의 차별성을 가지고 있습니다.  

## 2.3. 사회적 가치 도입 계획 
> 본 프로젝트는 단순히 기술적인 성과를 넘어 **사회적 가치** 창출을 지향합니다.  

- **공공성 확대**  
  아마추어 리그·청소년 클럽 등도 전술 분석의 기회를 가질 수 있도록 접근 장벽을 낮췄습니다. 이는 축구 저변 확대와 공정한 경쟁 환경 조성에 기여합니다.  
- **지속 가능성**  
  클라우드 및 경량화된 AI 모델을 사용함으로써 시스템 유지·운영에 필요한 리소스를 줄였고, 장기적으로 누구나 활용 가능한 지속 가능한 스포츠 분석 데이터를 제공합니다.  
  대규모 장비나 고사양 서버 없이도 운영이 가능하기 때문에, 불필요한 에너지 소비를 최소화할 수 있습니다. 데이터 전송과 저장 또한 클라우드 기반으로 최적화되어 있습니다.
- **교육적 가치**  
  청소년 선수들이 자신의 플레이 데이터를 쉽게 확인하고 학습에 활용할 수 있으며, 지도자들은 이를 통해 데이터 기반 피드백을 제공할 수 있습니다.  

# 3. 시스템 설계
## 3.1. 시스템 구성도
> 서비스 전체 라우팅 구조를 나타내는 구성도입니다.

<img width="940" height="506" alt="image" src="https://github.com/user-attachments/assets/f2140552-f663-4506-8787-4b6b3783230e" />


## 3.2. 사용 기술
> 프론트엔드, 백엔드, API 등 구체 기술 스택
- **FastAPI** : FastAPI는 현대적이고, 빠르며, 파이썬 표준 타입 힌트에 기초한 Python의 API를 빌드하기 위한 웹 프레임워크입니다. Node.js 및 Go와 대등할 정도로 매우 높은 처리 성능을 제공하고 있다. 매우 직관적으로 사용하기 쉽고, 코드 중복을 최소화하도록 설계되었다.
- **YOLOv8n** : YOLO (You Only Look Once)는 실시간 객체 검출 시스템으로, CNN(Convolutional Neural Networks) 딥러닝 모델을 기반으로 특징을 추출한 뒤 이를 이용해서 물체의 종류와 위치를 Bounding Box 로 표시해 Label로 분류한다.
- **웹 프레임워크** : 웹 프레임워크는 동적인 웹 페이지나 웹 애플리케이션을 효율적으로 개발할 수 있도록 구조와 기능을 제공하는 소프트웨어 도구이다. 웹 사이트를 처음부터 직접 구현하려면 HTTP 요청/응답 처리, 라우팅, 데이터베이스 연결, 보안 처리 등 복잡한 기능을 전부 개발해야 하는데, 프레임워크는 이를 미리 구현해두어 개발자가 핵심 기능에 집중할 수 있게 도와준다.
- **라우팅(Routing)** : 라우팅은 웹 애플리케이션에서 사용자가 요청한 URL을 특정 기능이나 함수와 연결해주는 과정을 의미한다. 이를 통해 경로에 따라 업로드, 분석 등 서로 다른 동작을 수행할 수 있다. 라우팅을 통해 사용자가 영상을 업로드하고 분석을 요청하는 기능을 손쉽게 구현할 수 있다.
- **MobileNet(모바일넷)** : MobileNet은 구글(Howard et al., 2017)에서 제안한 경량 딥러닝 신경망 구조로, 스마트폰이나 임베디드 기기처럼 계산 자원이 제한된 환경에서 효율적으로 동작할 수 있도록 설계되었다. 기존의 합성곱 신경망(CNN)은 높은 정확도를 보장하지만, 연산량이 많고 모델 크기가 커서 모바일 환경에서 활용하기 어렵다는 한계가 있었다. MobileNet은 이를 해결하기 위해 Depthwise Separable Convolution이라는 연산 방식을 도입하여 연산량과 파라미터 수를 획기적으로 줄였다.
- **LSTM(Long Short-Term Memory)** : LSTM(Long Short-Term Memory)은 1997년 Hochreiter와 Schmidhuber가 제안한 순환 신경망(Recurrent Neural Network, RNN)의 한 종류이다. 기존 RNN은 시계열 데이터나 연속적인 데이터를 처리하는 데 적합하지만, 긴 시퀀스를 다룰 때 기울기 소실(vanishing gradient) 문제와 장기 의존성(long-term dependency) 학습의 어려움이 발생한다. LSTM은 이러한 한계를 극복하기 위해 **메모리 셀(cell state)**과 게이트(gate) 구조를 도입하여, 중요한 정보는 오래 기억하고 불필요한 정보는 망각할 수 있도록 설계되었다.


# 4. 개발 결과
## 4.1. 전체 시스템 흐름도
> 기능 흐름 설명 및 도식화 가능
>
## 4.2. 기능 설명 및 주요 기능 명세서
> 주요 기능에 대한 상세 설명, 각 기능의 입력/출력 및 설명

- **프론트엔드** : 본 프로젝트의 프론트엔드는 단일 TML 문서 내에 정적 마크업(HTML), 스타일(CSS), 동적 행위(JavaScript)를 결합한 구조로 구현되었다. 배포 단순성과 재현성을 확보하였다. 핵심 목표는 사용자가 영상 파일을 쉽게 업로드하고, 분석 진행 상태를 확인하며, 결과 영상을 재생하고 전술적 포메이션을 확인할 수 있도록 하는데 있다.
  
- **화면 구조와 스타일** : 레이아웃은 좌측 사이드바와 우측 메인 영역으로 구성되어져 있다. 메인 영역은 3개의 파트로 구분되어지는데: 1. 영상 업로드, 2. 분석 설정, 3. 분석 결과로 구성되어져 있다. 우측 상단에는 ‘새 분석’ 버튼으로 영상 초기화하는 버튼이 있다. 업로드 박스는 drag & drop으로 하여 사용자가 편하게 업로드 할 수 있도록 하였다. 2번째인 분석 설정에서 영상에 대한 분석을 설정하고 “분석 시작” 버튼을 눌러 분석을 시작하고, 3번째 분석 결과에서 영상에 대한 분석 결과를 보여준다.
  <img width="940" height="462" alt="image" src="https://github.com/user-attachments/assets/d03326eb-379d-4a2f-bdb2-0fbde8b1c0f4" />
  <img width="940" height="388" alt="image" src="https://github.com/user-attachments/assets/1980d158-27dc-4fa0-a249-28ff83fd05f8" />
  <img width="940" height="312" alt="image" src="https://github.com/user-attachments/assets/caecb17a-a269-49c5-a96e-254117ee396f" />
  
- **선수 객체 탐지(YOLOv8n 모델 사용)** : 첫번째 단계는 Ultralytics의 YOLO모델을 사용하여 각 선수를 탐지하는 것 입니다. 각 선수를 보다 높은 정확도로 탐지하기 위해 선수만을 탐지하는 모델로 fine-tuning 작업을 진행하였습니다. Fine-tuning 작업을 위해 먼저 데이터 전처리 과정을 겨쳤습니다. 축구 영상을 프레임별로 나누고, 라벨링 파일(JSON형식)을 파싱하여 모델 학습용 라벨링(txt형식)으로 변환하였습니다. 이후 모델학습을 진행하였고, 탐지 개선도는 다음과 같습니다.

  | 구분              | 장면1 | 장면2 | 장면3 | 장면4 | 장면5 | 평균   |
  |-------------------|-------|-------|-------|-------|-------|--------|
  | 기본 YOLO 모델    | 6/20  | 11/19 | 18/22 | 17/20 | 15/21 | 70.1%  |
  | Fine-tuning된 YOLO 모델 | 18/19 | 22/22 | 20/20 | 21/21 | 21/21 | 98.9%  |

  약 28퍼센트 정확도 개선되었습니다.
  
  <img width="914" height="521" alt="image" src="https://github.com/user-attachments/assets/70208353-2a70-40e7-a04c-26e82a239e22" />
  Yolov8n 기본 모델 탐지 결과

  <img width="940" height="531" alt="image" src="https://github.com/user-attachments/assets/12a1d1c6-94c1-4343-bef7-0c94464db90b" />
  Fine-tuning된 모델 탐지 결과

- **MobileNet** : 이미지 분석에 사용되는 경량 CNN모델 MobileNet을 사용하여 각 팀을 식별했습니다.

YOLO 모델 하나로 선수 식별과 팀 분류를 동시에 하지 않은 이유는 다음과 같습니다.
1.	모델 분리로, 개선이 필요한 모델만 따로 학습시켜 비용을 절감 할 수 있다.
2.	YOLO모델 자체로 전부 학습을 시키면 이후 식별 시 한 화면에 3개 이상의 팀이 식별 될 수 있다.
3.	YOLO의 ByteTrack을 활용하여 Tracking을 시작할 때만 MobileNet을 사용해 팀을 분류하기 때문에 컴퓨팅 파워가 절약되고 속도가 빨라진다.
위 까닭으로 YOLO는 선수 식별, MobileNet은 팀 구별을 담당시켜 개발하였습니다.

MobileNet 대신 바운딩박스 내 유니폼 색깔 구별방법이나, 컬러 클러스터링을 활용한 구별 기법도 테스트해 보았으나 정확도가 너무 낮아 실패하였던 시행착오를 거쳤습니다. MobileNet 학습법은 다음과 같습니다.

먼저 영상에서 선수 바운딩 박스를 추출해내 팀 선수별 데이터를 준비하였습니다.
<img width="722" height="453" alt="image" src="https://github.com/user-attachments/assets/c0126c08-9972-4833-a316-6761b2d4c056" />
흰색 유니폼의 서울팀 선수
<img width="713" height="505" alt="image" src="https://github.com/user-attachments/assets/cb81c63e-1bd8-43f3-96a1-8e5ed0eca524" />
초록 유니폼의 전북팀 선수
<img width="717" height="501" alt="image" src="https://github.com/user-attachments/assets/1495557b-d3b2-40e0-a5bd-7458785d22e1" />
검은 옷의 심판

각팀 2000장 이상, 심판 600장을 MobileNet에 학습시켜 바운딩박스의 선수를 탐지하는 모델을 학습시켰습니다.
<img width="458" height="458" alt="image" src="https://github.com/user-attachments/assets/60705642-9a86-4732-b67b-1d43f69a33a5" />
총 10 Epoch을 거쳐 학습시켰습니다.

-**YOLO와 MobileNet 통합작업** : YOLO는 선수 탐지, MobileNet은 팀 분류 역할을 담당합니다.
하지만 YOLO의 predict 기능을 사용해서 객체를 탐지하고, 이를 각각 MobileNet에서 분류하게 되면 많은 컴퓨팅 자원을 소모하게 됩니다.
따라서 YOLO의 객체 트레킹 기능인 ByteTrack기능을 활용하여 한 번 탐지한 객체는 최대한 같은 객체로 인식하게끔 설정합니다.
ByteTracking이 동작하기 시작할 때 MobileNet을 통해 딱 한 번 탐지하여 컴퓨팅 자원의 소모를 획기적으로 줄였습니다.
0~60frame 분석 소요시간			  (단위: 초)
| 구분              | 소요시간 (초) |
|-------------------|---------------|
| Predict + 팀 분류 | 100.79        |
| Track + 팀 분류   | 17.68         |
약 5.7배 성능 개선

MobileNet사용 팀 분류 성능 측정표 (임의의 장면 수동 측정)
(단위: 팀과 선수가 옳게 식별됨/화면에 보이는 모든 선수)
| 장면 1 | 장면 2 | 장면 3 | 장면 4 | 장면 5 |
|--------|--------|--------|--------|--------|
| 11/22  | 18/22  | 17/21  | 17/20  | 17/18  |
약 60% 정확도

<img width="482" height="271" alt="image" src="https://github.com/user-attachments/assets/97dc24c5-07ee-4d23-b056-42d97779bf1c" />
MobileNet을 활용한 팀 식별 (장면 1)

-**LSTM을 활용한 전술예측** : 전술분석을 하기 위해 먼저 라벨링 데이터 전처리 과정을 거쳤습니다.
<img width="452" height="389" alt="image" src="https://github.com/user-attachments/assets/4a61dfd6-0984-4890-8ad0-d5e1eb44585b" />
프레임 내 선수 위치
<img width="438" height="389" alt="image" src="https://github.com/user-attachments/assets/94c4d143-6040-424e-b14d-1536a5982139" />
프레임 구간 간 사용 전술

두 csv 파일을 활용해 LSTM 모델을 학습시켰습니다.
<img width="774" height="432" alt="image" src="https://github.com/user-attachments/assets/71bc8e81-6d85-4221-9903-770f2c419b5b" />
LSTM모델 분석을 통해 왼쪽 위에 예측된 전략이 표시됩니다.


- **FastAPI** : 본 프로젝트에서 프론트엔드 웹사이트와 AI 분석 모델을 연결하기 위해 Python 기반의 FastAPI 프레임워크를 서버로 활용하였다. FastAPI는 코드의 가독성과 유지보수성을 높이는 동시에 자동 문서화(Swagger UI)를 지원하는 경량 웹 프레임워크이다. 이를 통해 직관적으로 설계하고, 프론트엔드와의 데이터 교환을 체계적으로 관리할 수 있다. 프로젝트에서 FastAPI을 활용하여 자동 문서화와 타입 기반 코드 검증을 통해 개발 효율성을 향상시켰다. 프론트엔드에서 상태 폴링 방식으로 요청하는 동안 서버가 효율적으로 상태를 업데이트하고, 결과 데이터를 반환할 수 있어 실시간성을 보장할 수 있었다. 이처럼 본 연구에서 FastAPI는 프론트엔드 – AI 모델 – 데이터베이스를 연결하는 중간 역할을 담당하며, RESTful 구조를 기반으로 안정적인 데이터 교환을 보장한다는 점에서 적합한 웹 프레임워크이다. 


## 4.3. 디렉토리 구조
>
## 4.4. 산업체 멘토링 의견 및 반영 사항
> 멘토 피드백과 적용한 사례 정리

# 5. 설치 및 실행 방법
>
## 5.1. 설치절차 및 실행 방법
> 설치 명령어 및 준비 사항, 실행 명령어, 포트 정보 등
## 5.2. 오류 발생 시 해결 방법
> 선택 사항, 자주 발생하는 오류 및 해결책 등

# 6. 소개 자료 및 시연 영상
## 6.1. 프로젝트 소개 자료
> PPT 등
## 6.2. 시연 영상
> 영상 링크 또는 주요 장면 설명

# 7. 팀 구성
## 7.1. 팀원별 소개 및 역할 분담
>
## 7.2. 팀원 별 참여 후기
> 개별적으로 느낀 점, 협업, 기술적 어려움 극복 사례 등

# 8. 참고 문헌 및 출처

```


# Template for Capstone
이 레파지토리는 학생들이 캡스톤 프로젝트 결과물을 위한 레파지토리 생성시에 참고할 내용들을 담고 있습니다.
1. 레파지토리 생성
2. 레파지토리 구성
3. 레파지토리 제출 
4. README.md 가이드라인
5. README.md 작성팁

---

## 1. 레파지토리 생성
- [https://classroom.github.com/a/nRcUn8vA](https://classroom.github.com/a/nRcUn8vA)
- 위 Github Classroom 링크에 접속해 본인 조의 github 레파지토리를 생성하세요.

<img width="700" alt="깃헙 클래스룸 레포 생성" src="https://github.com/user-attachments/assets/34ca1f43-c2cd-4880-a39e-0dafd889c35f" />

- 레포지토리 생성 시 팀명은 `TEAM-{조 번호}` 형식으로 생성하세요.
- 예를 들어, 2025년도 3조의 팀명은 `TEAM-03` 입니다.
- 이 경우 `Capstone2025-team-03`이란 이름으로 레파지토리가 생성됩니다.

---

## 2. 레파지토리 구성
- 레파지토리 내에 README.md 파일 생성하고 아래의 가이드라인과 작성팁을 참고하여 README.md 파일을 작성하세요. (이 레파지토리의 SAMPLE_README.md 참조)
- 레파지토리 내에 docs 디렉토리를 생성하고 docs 디렉토리 내에는 과제 수행 하면서 작성한 각종 보고서, 발표자료를 올려둡니다. (이 레파지토리의 docs 디렉토리 참조)
- 그 밖에 레파지토리의 폴더 구성은 과제 결과물에 따라 자유롭게 구성하되 가급적 코드의 목적이나 기능에 따라 디렉토리를 나누어 구성하세요.

---

## 3. 레파지토리 제출 

- **`[주의]` 레파지토리 제출**은 해당 레파지토리의 ownership을 **학과 계정**으로 넘기는 것이므로 되돌릴 수 없습니다.
- **레파지토리 제출** 전, 더 이상 수정 사항이 없는지 다시 한번 확인하세요.
- github 레파지토리에서 Settings > General > Danger zone > Transfer 클릭
  <img src="https://github.com/user-attachments/assets/cb2361d4-e07e-4b5d-9116-aa80dddd8a8b" alt="소유주 변경 경로" width="500" />
  
- [ Specify an organization or username ]에 'PNUCSE'를 입력하고 확인 메세지를 입력하세요.
  <img src="https://github.com/user-attachments/assets/7c63955d-dcfe-4ac3-bdb6-7d2620575f3a" alt="소유주 변경" width="400" />

---

## 4. README.md 가이드 라인
- README 파일 작성시에 아래의 5가지 항목의 내용은 필수적으로 포함해야 합니다.
- 아래의 항목이외에 프로젝트의 이해를 돕기 위한 내용을 추가해도 됩니다.
- SAMPLE_README.md 이 단순한 형태의 예제이니 참고하세요.

```markdown
### 1. 프로젝트 배경
#### 1.1. 국내외 시장 현황 및 문제점
> 시장 조사 및 기존 문제점 서술

#### 1.2. 필요성과 기대효과
> 왜 이 프로젝트가 필요한지, 기대되는 효과 등

### 2. 개발 목표
#### 2.1. 목표 및 세부 내용
> 전체적인 개발 목표, 주요 기능 및 기획 내용

#### 2.2. 기존 서비스 대비 차별성 
> 유사 서비스 비교 및 차별점 부각

#### 2.3. 사회적 가치 도입 계획 
> 프로젝트의 공공성, 지속 가능성, 환경 보호 등
### 3. 시스템 설계
#### 3.1. 시스템 구성도
> 이미지 혹은 텍스트로 시스템 아키텍쳐 작성
>
#### 3.2. 사용 기술
> 프론트엔드, 백엔드, API 등 구체 기술 스택

### 4. 개발 결과
#### 4.1. 전체 시스템 흐름도
> 기능 흐름 설명 및 도식화 가능
>
#### 4.2. 기능 설명 및 주요 기능 명세서
> 주요 기능에 대한 상세 설명, 각 기능의 입력/출력 및 설명
>
#### 4.3. 디렉토리 구조
>
#### 4.4. 산업체 멘토링 의견 및 반영 사항
> 멘토 피드백과 적용한 사례 정리

### 5. 설치 및 실행 방법
>
#### 5.1. 설치절차 및 실행 방법
> 설치 명령어 및 준비 사항, 실행 명령어, 포트 정보 등
#### 5.2. 오류 발생 시 해결 방법
> 선택 사항, 자주 발생하는 오류 및 해결책 등

### 6. 소개 자료 및 시연 영상
#### 6.1. 프로젝트 소개 자료
> PPT 등
#### 6.2. 시연 영상
> 영상 링크 또는 주요 장면 설명

### 7. 팀 구성
#### 7.1. 팀원별 소개 및 역할 분담
>
#### 7.2. 팀원 별 참여 후기
> 개별적으로 느낀 점, 협업, 기술적 어려움 극복 사례 등

### 8. 참고 문헌 및 출처

```

## 5. README.md 작성팁 
* 마크다운 언어를 이용해 README.md 파일을 작성할 때 참고할 수 있는 마크다운 언어 문법을 공유합니다.  
* 다양한 예제와 보다 자세한 문법은 [이 문서](https://www.markdownguide.org/basic-syntax/)를 참고하세요.

### 5.1. 헤더 Header
```
# This is a Header 1
## This is a Header 2
### This is a Header 3
#### This is a Header 4
##### This is a Header 5
###### This is a Header 6
####### This is a Header 7 은 지원되지 않습니다.
```
<br />

### 5.2. 인용문 BlockQuote
```
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.
```
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.
<br />

### 5.3. 목록 List
* **Ordered List**
```
1. first
2. second
3. third  
```
1. first
2. second
3. third
<br />

* **Unordered List**
```
* 하나
  * 둘

+ 하나
  + 둘

- 하나
  - 둘
```
* 하나
  * 둘

+ 하나
  + 둘

- 하나
  - 둘
<br />

### 5.4. 코드 CodeBlock
* 코드 블럭 이용 '``'
```
여러줄 주석 "```" 이용
"```
#include <stdio.h>
int main(void){
  printf("Hello world!");
  return 0;
}
```"

단어 주석 "`" 이용
"`Hello world`"

* 큰 따움표(") 없이 사용하세요.
``` 
<br />

### 5.5. 링크 Link
```
[Title](link)
[부산대학교 정보컴퓨터공학부](https://cse.pusan.ac.kr/cse/index..do)

<link>
<https://cse.pusan.ac.kr/cse/index..do>
``` 
[부산대학교 정보컴퓨터공학부](https://cse.pusan.ac.kr/cse/index..do)

<https://cse.pusan.ac.kr/cse/index..do>
<br />

### 5.6. 강조 Highlighting
```
*single asterisks*
_single underscores_
**double asterisks**
__double underscores__
~~cancelline~~
```
*single asterisks* <br />
_single underscores_ <br />
**double asterisks** <br />
__double underscores__ <br />
~~cancelline~~  <br />
<br />

### 5.7. 이미지 Image
```
<img src="image URL" width="600px" title="Title" alt="Alt text"></img>
![Alt text](image URL "Optional title")
```
- 웹에서 작성한다면 README.md 내용 안으로 이미지를 드래그 앤 드롭하면 이미지가 생성됩니다.
- 웹이 아닌 로컬에서 작성한다면, github issue에 이미지를 드래그 앤 드롭하여 image url 을 얻을 수 있습니다. (URL만 복사하고 issue는 제출 안 함.)
  <img src="https://github.com/user-attachments/assets/0fe3bff1-7a2b-4df3-b230-cac4ef5f6d0b" alt="이슈에 image 올림" width="600" />
  <img src="https://github.com/user-attachments/assets/251c6d42-b36b-4ad4-9cfa-fa2cc67a9a50" alt="image url 복사" width="600" />


### 5.8. 유튜브 영상 추가
```markdown
[![영상 이름](유튜브 영상 썸네일 URL)](유튜브 영상 URL)
[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://www.youtube.com/watch?v=zh_gQ_lmLqE)    
```
[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://www.youtube.com/watch?v=zh_gQ_lmLqE)    

- 이때 유튜브 영상 썸네일 URL은 유투브 영상 URL로부터 다음과 같이 얻을 수 있습니다.

- `Youtube URL`: https://www.youtube.com/watch?v={동영상 ID}
- `Youtube Thumbnail URL`: http://img.youtube.com/vi/{동영상 ID}/0.jpg 
- 예를 들어, https://www.youtube.com/watch?v=zh_gQ_lmLqE 라고 하면 썸네일의 주소는 http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg 이다.

