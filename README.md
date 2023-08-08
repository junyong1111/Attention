# Attention
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

- ****Transformer : Attention Is All You Need****
    

### RNN (Recurrent Neural Network):

RNN은 순차적인 데이터를 처리하는 데 사용되는 신경망 구조이다. RNN은 이전 시간 단계의 입력을 현재 시간 단계의 입력과 함께 처리하여 순차적인 정보를 유지하고 활용할 수 있고 텍스트, 음성 등 순차적인 시계열 데이터 처리에 유용하다

- 특징: 순차적인 데이터 처리, 이전 상태의 정보를 기억
- 장점: 순차적인 패턴을 학습할 수 있음, 시계열 데이터 처리에 적합
- 단점: 장기 의존성(Long-Term Dependency)을 잘 학습하지 못하는 문제, Gradient Vanishing/Exploding 등의 문제 발생

<img width="553" alt="1" src="https://github.com/junyong1111/Attention/assets/79856225/11058141-d431-45f9-bb99-e04569a06bda">

각각의 단어를 컴퓨터가 이해할 수 있도록 Vector로 변환하며 이것을 **워드 임베딩**이라고 한다

일반적으로 사용되는 워드 임베딩 기법에는 다음과 같은 것들이 있다.

- **Word2Vec**: 2013년에 구글 연구원인 Tomas Mikolov 등에 의해 제안된 기법으로, CBOW(Continuous Bag of Words)와 Skip-gram 두 가지 모델이 있다. CBOW는 주변 단어들을 통해 중심 단어를 예측하는 방식으로 학습하며, Skip-gram은 중심 단어를 통해 주변 단어들을 예측하는 방식으로 학습한다.
- **GloVe(Gloabal Vectors for Word Representation)**: 스탠포드 대학의 연구자들이 제안한 기법으로, Word2Vec과 유사한 방식으로 단어의 출현 빈도와 동시 등장 확률 통계를 이용하여 단어들을 벡터로 변환한다.
1. **FastText**: 페이스북 AI 연구팀이 개발한 기법으로, Word2Vec의 확장된 버전이다. 단어를 더 작은 서브워드(subword) 단위로 분해하여 임베딩하고, 이를 통해 희소한 단어들에 대한 표현을 더욱 효과적으로 학습할 수 있다.
2. **ELMo(Embeddings from Language Models)**: 사전 훈련된 언어 모델을 활용하여 단어 임베딩을 학습하는 기법으로, 문맥 정보를 고려하여 단어의 다의성과 문맥 의존성을 잘 반영한다.
3. **BERT(Bidirectional Encoder Representations from Transformers)**: 구글 AI 연구팀이 제안한 기법으로, 양방향 Transformer 인코더를 사용하여 문장 내 단어들을 임베딩한다. BERT는 현재 자연어 처리 분야에서 가장 성능이 우수한 워드 임베딩 기법 중 하나로 인정받고 있다.

### LSTM

LSTM은 RNN의 단점 중 하나인 장기 의존성 문제를 해결하기 위해 제안된 변형된 RNN 구조이다. LSTM은 시간적인 의존성을 잘 다룰 수 있도록 설계되었다. LSTM은 게이트를 이용하여 특정 시간 단계에서 중요한 정보를 기억하고, 필요에 따라 이를 장기적으로 전달하거나 삭제할 수 있다.

- 특징: 장기 의존성을 다루기 위한 메모리 셀, 입력 게이트, 삭제 게이트, 출력 게이트 등의 구조
- 장점: 장기 의존성 문제를 해결, 시계열 데이터 처리에 적합
- 단점: 많은 파라미터와 연산이 필요하여 학습과정이 복잡함, 계산량이 크고 처리 속도가 상대적으로 느릴 수 있음

![2](https://github.com/junyong1111/Attention/assets/79856225/5778db32-8512-42ed-80c9-aa5019af3789)

### **Seq2Seq(Sequence-to-Sequence)**

딥러닝 모델 중 하나로, 시퀀스 데이터를 입력으로 받아 다른 시퀀스 데이터를 출력으로 생성하는 모델이다. 주로 자연어 처리(Natural Language Processing, NLP) 분야에서 사용되며, 기계 번역, 챗봇, 요약, 질의응답 등의 다양한 작업에 적용된다.

Seq2Seq 모델은 기본적으로 두 개의 RNN(Recurrent Neural Network)을 활용하여 동작한다

1. **인코더(Encoder)**: 입력 시퀀스를 받아 **고정 길이의 문맥 벡터를 생성한다**. 인코더 RNN은 입력 시퀀스의 각 단어를 순차적으로 입력받고, 중간 은닉 상태(hidden state)를 업데이트하여 시퀀스 정보를 요약한다. 인코더의 마지막 은닉 상태를 문맥 벡터(context vector)로 사용한다. 이 문맥 벡터는 입력 시퀀스의 정보를 압축하고 다른 RNN으로 전달한다.
2. **디코더(Decoder)**: 인코더가 생성한 문맥 벡터를 초기 상태로 하여 출력 시퀀스를 생성한다. 디코더 RNN은 시작 토큰(예: `<start>`)을 입력으로 받아 첫 번째 단어를 생성하고, 그 다음에는 이전 단어를 입력으로 받아 다음 단어를 예측한다. 이런 식으로 디코더는 단어 단위로 순차적으로 출력 시퀀스를 생성한다. 디코더는 끝 토큰(예: `<end>`)을 생성할 때까지 단어를 계속해서 생성하고, 최종적으로 생성된 시퀀스를 출력으로 반환한다.

Seq2Seq 모델은 훈련과 추론(테스트) 단계에서 다르게 동작한다

- 훈련: 훈련 데이터의 입력 시퀀스와 출력 시퀀스를 사용하여 인코더와 디코더를 동시에 학습한다. 손실 함수로는 보통 교차 엔트로피 손실(Cross-Entropy Loss)을 사용하여 예측된 출력 시퀀스와 실제 출력 시퀀스의 차이를 최소화한다.
- 추론: 훈련된 Seq2Seq 모델을 사용하여 새로운 입력 시퀀스에 대한 출력 시퀀스를 생성한다.  디코더는 시작 토큰을 입력으로 받아 다음 단어를 예측하고, 이를 반복하여 출력 시퀀스를 생성한다. 일반적으로 빔 서치(Beam Search) 등의 방법을 사용하여 더 나은 출력 시퀀스를 찾는다.

Seq2Seq 모델은 자연어 처리 태스크에서 탁월한 성능을 보여주고, 다양한 변형 및 발전된 모델들이 제안되어 계속해서 연구되고 있다. 이러한 모델들은 자연어 이해와 생성 작업에 큰 도움을 주며, 실제 응용에서도 많이 활용되고 있다.

<img width="553" alt="3" src="https://github.com/junyong1111/Attention/assets/79856225/772bf6da-6238-4f0c-95ab-fd13b2995903">


<img width="1095" alt="4" src="https://github.com/junyong1111/Attention/assets/79856225/37133b0a-ae34-432e-a1c9-d8373fe0d469">


## Transformer

### Architecture

<img width="1164" alt="5" src="https://github.com/junyong1111/Attention/assets/79856225/6177b3d5-911c-48f3-ae7c-ff56964f3761">

<img width="1096" alt="6" src="https://github.com/junyong1111/Attention/assets/79856225/71b89fff-14cf-422d-811b-6120e2b96b14">

### 1. Word Embedding

<img width="234" alt="7" src="https://github.com/junyong1111/Attention/assets/79856225/64d03bce-577a-4f9f-ba7a-61970cfbf951">


**각각의 단어들을 벡터로 변환 하며 벡터들이 한번에 들어가므로 위치 정보가 없다.** 

**따라서 위치값을 알려주기 위해 Positional Encoding을 해준다.**

<img width="557" alt="8" src="https://github.com/junyong1111/Attention/assets/79856225/f2c5cb78-3c28-4501-bdd2-f8718ce47230">

<img width="511" alt="9" src="https://github.com/junyong1111/Attention/assets/79856225/d4db980d-9250-427a-8a9f-0146fbfc6fc4">

### 2. Self-Attention

<img width="212" alt="10" src="https://github.com/junyong1111/Attention/assets/79856225/6b5922c5-7c4e-4ee5-865c-9cb12c259808">

Self-Attention은 문장 내 단어들 간의 상호 관계를 이해하기 위한 메커니즘이다. 각 단어는 모든 다른 단어들에 대해 가중치를 계산하여 중요한 정보를 집중적으로 수집한다. 이 가중치는 "어텐션 스코어"로 나타내며, 주어진 단어와 다른 단어 간의 유사도를 기반으로 계산된다. 이러한 Self-Attention을 여러 차례 반복하여 문장 내 정보를 효과적으로 추출한다.

**어텐션 스코어를 계산하는 방법**

1. Q**uery, Key, Value 생성:** 주어진 입력 시퀀스에 대해 Query, Key 및 Value를 생성한다. 이들은 선형 변환을 통해 생성되며, Query는 주로 해당 위치에서 어떤 정보를 얻고자 하는지를 나타내고, Key와 Value는 각각 해당 위치의 정보를 나타낸다.
2. **어텐션 스코어 계산:** Query와 Key 사이의 유사도를 계산한다. 이를 위해 Query와 Key 사이의 내적(dot product)을 계산한다. 내적의 결과는 어텐션 스코어가 되며, 스코어는 각 Key에 대해 Query와의 유사성을 나타낸다.
3. **스케일링:** 어텐션 스코어가 너무 크면 계산이 불안정해질 수 있으므로, 스케일링을 적용한다. 일반적으로 스케일링 인자로 루트값을 사용하여 어텐션 스코어를 나누어준다.
4. **소프트맥스 적용:** 스케일링된 어텐션 스코어에 소프트맥스 함수를 적용하여 정규화된 어텐션 가중치를 얻는다. 이렇게 하면 각 Key의 중요도가 확률 분포로 표현된다.
5. **가중합 계산:** 소프트맥스를 통해 얻은 어텐션 가중치와 Value를 곱하여 가중합을 계산한다. 이 가중합은 Query 위치에서의 최종 출력을 생성하는 데 사용된다.

찾고자하는 단어의 Query를 가지고 모든 문장의 Key를 내적하여 가장 유사한 값을 찾아낸다.

<img width="512" alt="11" src="https://github.com/junyong1111/Attention/assets/79856225/fa89ac5d-77fa-4488-aeaa-6b9819509749">

<img width="534" alt="12" src="https://github.com/junyong1111/Attention/assets/79856225/3e6d81b1-a0d5-4d80-861a-ecc9d9ac3fd4">

### 3. 정리

1. 워드 임베딩이 딥러닝이 생성한 Weight와 연산하여 Query, Key, Value 3개의 벡터로 변환
    
    <img width="984" alt="13" src="https://github.com/junyong1111/Attention/assets/79856225/e48ba6a0-0388-46b8-964c-8b93735c36e2">

    
2. 각 해당하는 Query벡터는 순차적으로 모든 Key와 내적 연산 후 소프트맥스 연산
    
    <img width="1113" alt="14" src="https://github.com/junyong1111/Attention/assets/79856225/98938fad-ea08-4482-9033-0cf3bf826273">
    
3. 소프트맥스 결과값과 Value 값과 곱한 후 모든 값들을 합쳐서 하나의output을 생성
    
    <img width="989" alt="15" src="https://github.com/junyong1111/Attention/assets/79856225/c03c8414-c3ed-4b83-a6de-25c809347ed0">
    
4. 위와 같은 방법으로 각각의 워드임베딩마다 각각의 output을 생성
    
    <img width="963" alt="16" src="https://github.com/junyong1111/Attention/assets/79856225/c5522082-4b02-41c8-9ae9-a9460e0ead60">
    
5. 문장을 input으로 넣으면 해당하는 문장을 나타내는 output 행렬이 나옴
    
    <img width="879" alt="17" src="https://github.com/junyong1111/Attention/assets/79856225/b9715632-45ad-4a3a-80ce-8d4c5c456ed4">

    
6. 각각의 어텐션에서 나온 벡터들을 concat 이후 학습된 weight와 연산하여 초기 demention을 맞춰줌
    
    <img width="1031" alt="18" src="https://github.com/junyong1111/Attention/assets/79856225/28abe395-b6a6-46b1-9194-5922001053cb">
    
    1. 전체 과정
        
        <img width="1025" alt="19" src="https://github.com/junyong1111/Attention/assets/79856225/288a2110-1ce8-4f76-8e19-726fb0efcf46">