# 확산 모델의 작동방식에 대하여
#### 1학년 4반 1번 권부성

## 1. 위 주제를 선정하게 된 이유
저는 평소에 컴퓨터와 코딩, 작업의 자동화에 대해 관심이 있고, 특히 생성형 AI에 대해 큰 관심을 가졌기 때문에 위 주제를 선정하였습니다.



## 2. 탐구의 실현
Stable Diffusion에 대해 분석.

Stable Diffusion은 stability AI이라는 단체가 만들었습니다.
LMU 뮌헨의 CompVis 그룹에서 개발한 LDM(잠재 확산 모델)이라는 일종의 확산 모델(DM)을 기반으로 만들어졌습니다.

Stable Diffusion은 “High-Resolution Image Synthesis with Latent Diffusion Models”의 논문에 기반한 GAN(생성형 인공지능)입니다. GAN은 기존의 전통적 AI와 달리 분석에 그치지 않고 분석후, 이미지, 요약등 다양한 기능을 수행할 수 있는 AI입니다.
따러서, Stable Diffusion은 입력을 분석하여 그것에 대한 출력을 통해 멀티모달을 생성해냅니다.
텍스트를 영상화, 애니매이션화, 동영상화를 주목적으로 하고 있습니다.
"LAION-5B"이라는 현존 가장 거대(large)하고 자유롭게 엑세스 할 수 있는 멀티-모달데이터 세트를 통해
50억개 이상의 텍스트와 512x512의 해상도를 가진 이미지로 훈련 되었습니다.

Stable Diffusion은 전이 학습을 통해 최소 5개의 이미지를 특정 조건이 충족하도록 미세 조정할 수 있습니다.
전이학습이란 한 분야의 문제를 해결하기 위해서 얻은 정보를 다른 문제를 푸는데 사용하는 방식입니다.
기계의 시각적 이해를 목표로 하는 컴퓨터 비전의 영역에서 전이 학습으로 수행된 모델들이 높은 성능을 보이고 있어
컴퓨터 비전 관련 분야에서 가장 많이 사용되는 방법 중에 하나입니다.

Stable Diffusion은 다른 모델(DALL-E등)과 달리 이미지의 픽셀의 영역이 아닌 이미지의 픽셀의 잠재공간에서 활동합니다.
때문에, 다른 모델과 달리 속도가 빠르고 메모리의 소모가 덜 합니다.


Stable Diffusion은 크게 3가지의 요소로 구성되어 있습니다.
1. 사전 훈련된 문자 인코더(text encoder)
2. UNet 잡음 예측기(UNet noise predictor)
3. 변형 오토 엔코더-디코더 모델(Variational autoencoder-decoder model)

사전 훈련된 문자 인코더(text encoder)는 미리 훈련된 거대 트랜스포머 언어 모델(large pretrained transformer language model)입니다.
트랜스포머란 어떠한 문자가 주어질때 다음 문자를 확율적으로 예측하는 딥러닝 모델중 하나 입니다.
예를 들어, "나는" 이라는 문자가 주어질때 "학생이다", "선생이다"등 훈련된 데어터 셋중 가장 높은 확율을 가진 문자를 도출합니다.
즉, 문자 인코더는 위와 같은 방식을 가집니다.

Stable Diffusion은 문자 인코딩을 위해 CLIP(Contrastive Language-Image Pre-training model)의 사전 훈련된 문자 인코더의 일부분을 사용합니다.
표제(간단한 제목)를 입력으로 하고 77×768 차원 토큰 임베딩을 도출합니다.
토큰의 개수가 77개인 이유는 75개가 표제의 문자 토큰이며, 1개는 시작 토큰, 1개는 종료 토큰이기 때문입니다.

변형된 자동 엔코더-디코더 모델(Variational autoencoder-decoder model)(이하 VAE)은 두가지의 작업을 합니다.
첫번째로 VAE는 인코더로서 원본 이미지의 픽셀의 잠재 공간을 생성합니다.
두번째로 VAE는 디코더로서 텍스트의 조건에 따른 잠재 공간에서 이미지를 예측합니다.

UNet 잡음 예측기(UNet noise predictor)(이하 UNet)는 잠재 공간에서 노이즈를 예측합니다.
UNet은 이미지가 노이즈가 될때까지 다운샘플링을 합니다. 즉, 잠재 공간을 도출합니다.
잠재 공간에서 일련의 과정을 거쳐 예측된 노이즈를 UNet이 업램플링 하여 디코더로서의 기능을 수행합니다.
UNet은 디코더로서의 기능을 수행하는 과정 중 "문자의 조건화"와 잠재 공간의 데이터를 변형하는 작업을 수행합니다.
"문자의 조건화"란 표제 정보를 잠재 공간에 추가하는 과정입니다.
또한, UNet은 잠재적 공간에서만 작동하고 문자의 표제에 따라 원본 이미지의 픽셀의 잠개공간의 데이터를 변형하지 않을 수 있습니다.

문자 인코더의 출력물과 노이즈를 합쳐, 조건화된 문자의 노이즈 벡터를 만듭니다.
UNet이 조건화된 문자의 노이즈 벡터에서 이에 맞는 노이즈를 예측합니다.
즉, 예측된 잠재 공간을 도출합니다.

마지막으로 VAE가 예측된 이미지의 잠재공간을 디코딩 함으로써 최종적인 데이터를 도출해냅니다.


## 3. 탐구를 하며

위 주제를 탐구하면서 Stable Diffusion과 같은 Diffuison model의 작동방식을 알고 이해도를 높여서 AI에 대해 오해하고 있는 지식을 보강하였습니다.
자료를 조사하면서 VAE, latent space과 같은 다양한 단어를 접했습니다. 이를통해 AI에 관한 기술적 지식을 더 넓혔습니다.
위와같은 이유로 AI에 대해 더 자세히 알았고, AI에 관해 높은 수준의 지식을 요구하는 기술적 설명에 대해 더 이상 피하지 않고 그것을 이해할 수 있었습니다.
 
다음에 이와 관련된 주제를 또 탐구하게 된다면, 로컬 환경에서 Stable Diffusion을 실행시킬 것이고 로컬 환경에서 각각 머신러닝을 수행시키고 적대적 공격을 수행하여 같은 인풋에 결과물(아웃풋)이 어떻게 달라지는지 탐구해보고 싶다.


## 세부 특기 내용:

생성형 AI에 관심이 있어 자율 주제 탐구 프로젝트(2023.10.16.~2023.11.15.)에 참여하여 “확산 모델의 작동방식에 대하여“를 주제로 Stable Diffiusion의 토대가 되는 논문들을 참고하였고 인터넷 검색을 통하여 작동방식에 대한 정보를 모아 탐구를 실행하였으며, 앞으로 로컬환경에서 모델을 구현하고 적대적 공격과 같은 컴퓨터 비전으로서 치명적인 공격이 어떻게 결과물에 영향을 미치는지 탐구해보고 싶어함.


## 참고자료:
<https://www.techtube.co.kr/news/articleView.html?idxno=2791>

<https://huggingface.co/blog/stable_diffusion>


<https://learnopencv.com/stable-diffusion-generative-ai>


<https://dacon.io/forum/405988>


<https://aws.amazon.com/ko/what-is/stable-diffusion>


<https://arxiv.org/abs/2112.10752>


<https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=5zrjyyxpW8px>


