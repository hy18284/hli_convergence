#! /bin/bash

for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python run_training.py \
        --config configs/run_peld_training_config.yml \
        --trainer.logger.init_args.name peld_${rate} \
        --data.init_args.limit_train $rate \
        --model.personality false \
        --model.emotion true \
        --model.sentiment true
done 