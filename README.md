# hli_convergence

# 설정

## 도커 설치

[https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

## wandb.ai 회원가입

[https://wandb.ai/](https://wandb.ai/)

## 코드 받기 및 컨테이너 실행

```bash
git clone git@github.com:hy18284/hli_convergence.git
cd hli_convergence
bash bash scripts/docker_run_demo.sh
docker attach hy_conv_demo
cd /root/conv
```

## 디펜던시 설치

```bash
bash scripts/install_dependencies.sh apt-get
```

엔터 -> yes -> 엔터 -> yes

```bash
exec bash
```

## 콘다 환경 초기화

```bash
conda env create -f environment.yml
conda activate conv
```

# 학습 실행

## PELD 감정 및 펄스널리티 트레이트 예측 학습 실행

```bash
python run_training.py --config configs/run_peld_emo_person_training_config copy.yml
```

## PELD 감정 예측 학습 실행

```bash
python run_training.py --config configs/run_peld_emo_person_training_config copy.yml
```

## FriendsPersona 학습 실행

```bash
python run_training.py --config configs/run_fp_training_config.yml
```

## 학습 결과 확인 

1. [https://wandb.ai/](https://wandb.ai/)
2. Projects -> hli_conv