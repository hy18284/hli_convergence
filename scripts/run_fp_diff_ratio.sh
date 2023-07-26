#! /bin/bash

for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 null
do
    python run_training.py \
        --config configs/run_fp_training_config.yml \
        --data.init_args.limit_train $rate \
        --data.init_args.seed 43 \
        --trainer.logger.init_args.name friends_persona_${rate} \
        --trainer.max_steps 2000
done 