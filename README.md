# Attention
Transformer : Attention Is All You Need

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

- ****Transformer : Attention Is All You Need****
    
    [1706.03762.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1a8de203-3171-4106-87f5-c13ff1a2e4c3/1706.03762.pdf)
    

### RNN (Recurrent Neural Network):

RNN은 순차적인 데이터를 처리하는 데 사용되는 신경망 구조이다. RNN은 이전 시간 단계의 입력을 현재 시간 단계의 입력과 함께 처리하여 순차적인 정보를 유지하고 활용할 수 있고 텍스트, 음성 등 순차적인 시계열 데이터 처리에 유용하다

- 특징: 순차적인 데이터 처리, 이전 상태의 정보를 기억
- 장점: 순차적인 패턴을 학습할 수 있음, 시계열 데이터 처리에 적합
- 단점: 장기 의존성(Long-Term Dependency)을 잘 학습하지 못하는 문제, Gradient Vanishing/Exploding 등의 문제 발생

![스크린샷 2023-08-07 오후 5.15.50.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bd122d0b-89c2-4d3f-bd57-eb2d1b2ece98/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-08-07_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.15.50.png)

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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46d031a9-2fbc-4c34-8424-9a45b16bd117/Untitled.png)

**Seq2Seq(Sequence-to-Sequence)**

딥러닝 모델 중 하나로, 시퀀스 데이터를 입력으로 받아 다른 시퀀스 데이터를 출력으로 생성하는 모델이다. 주로 자연어 처리(Natural Language Processing, NLP) 분야에서 사용되며, 기계 번역, 챗봇, 요약, 질의응답 등의 다양한 작업에 적용된다.

Seq2Seq 모델은 기본적으로 두 개의 RNN(Recurrent Neural Network)을 활용하여 동작한다

1. **인코더(Encoder)**: 입력 시퀀스를 받아 **고정 길이의 문맥 벡터를 생성한다**. 인코더 RNN은 입력 시퀀스의 각 단어를 순차적으로 입력받고, 중간 은닉 상태(hidden state)를 업데이트하여 시퀀스 정보를 요약한다. 인코더의 마지막 은닉 상태를 문맥 벡터(context vector)로 사용한다. 이 문맥 벡터는 입력 시퀀스의 정보를 압축하고 다른 RNN으로 전달한다.
2. **디코더(Decoder)**: 인코더가 생성한 문맥 벡터를 초기 상태로 하여 출력 시퀀스를 생성한다. 디코더 RNN은 시작 토큰(예: `<start>`)을 입력으로 받아 첫 번째 단어를 생성하고, 그 다음에는 이전 단어를 입력으로 받아 다음 단어를 예측한다. 이런 식으로 디코더는 단어 단위로 순차적으로 출력 시퀀스를 생성한다. 디코더는 끝 토큰(예: `<end>`)을 생성할 때까지 단어를 계속해서 생성하고, 최종적으로 생성된 시퀀스를 출력으로 반환한다.

Seq2Seq 모델은 훈련과 추론(테스트) 단계에서 다르게 동작한다

- 훈련: 훈련 데이터의 입력 시퀀스와 출력 시퀀스를 사용하여 인코더와 디코더를 동시에 학습한다. 손실 함수로는 보통 교차 엔트로피 손실(Cross-Entropy Loss)을 사용하여 예측된 출력 시퀀스와 실제 출력 시퀀스의 차이를 최소화한다.
- 추론: 훈련된 Seq2Seq 모델을 사용하여 새로운 입력 시퀀스에 대한 출력 시퀀스를 생성한다.  디코더는 시작 토큰을 입력으로 받아 다음 단어를 예측하고, 이를 반복하여 출력 시퀀스를 생성한다. 일반적으로 빔 서치(Beam Search) 등의 방법을 사용하여 더 나은 출력 시퀀스를 찾는다.

Seq2Seq 모델은 자연어 처리 태스크에서 탁월한 성능을 보여주고, 다양한 변형 및 발전된 모델들이 제안되어 계속해서 연구되고 있다. 이러한 모델들은 자연어 이해와 생성 작업에 큰 도움을 주며, 실제 응용에서도 많이 활용되고 있다.

### GRU