#! /bin/bash

docker run \
--name hy_conv_demo \
-it \
-v $(pwd):/root/conv \
--network host \
--gpus all \
--ipc host \
--privileged \
nvcr.io/nvidia/pytorch:23.01-py3 \
/bin/bash

