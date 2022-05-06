# language_model
 
언어 모델 구현 및 학습 코드.

대부분의 코드는

https://github.com/openspeech-team/openspeech

https://github.com/codertimo/BERT-pytorch

에서 따와서 약간의 수정을 거쳤습니다.

토크나이저의 라벨은 openspeech의 kspon_character를 사용했고,

데이터셋은 aihub의 음성 데이터셋 텍스트 파일들("한국어 음성", "한국어 강의 음성", "자유대화 음성(일반남녀)")을 사용했습니다.

# 학습 방법

0) 레포 설치
 ```bash
 $ git clone https://github.com/intrandom5/training-BERT-GPT
 $ cd training-BERT-GPT
 $ pip install -r requirements.txt
 ```

1) 데이터셋 준비

 dataset.txt
 
 ```bash
 문장을 한 줄 씩 작성해 주세요.
 문장을 한 줄 씩 작성해 주세요.
 문장을 한 줄 씩 작성해 주세요.
 ```
 
2) label 파일 생성

 ```bash
 generate_labels.py --text_file_path dataset.txt --csv_file_path labels.csv
 ```
 
3) 모델 학습 및 테스트

 모델은 train_model.ipynb로 학습해볼 수 있고, tensorboard에서 loss와 perplexity를 볼 수 있습니다.
 
 ```bash
 tensorboard --logdir runs
 ```
 
 모델을 저장한 후에 test_model.ipynb로 모델을 테스트 해볼 수 있습니다.
 
