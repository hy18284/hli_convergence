#! /bin/bash

docker run \
--name hy_conv \
-it \
-v ~/.ssh/id_rsa:/root/.ssh/id_rsa \
-v ~/.tmux.conf:/root/.tmux.conf \
-v $(pwd):/root/conv \
--network host \
--gpus all \
--ipc host \
--privileged \
nvcr.io/nvidia/pytorch:23.01-py3 \
/bin/bash

