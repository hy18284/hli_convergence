#! /bin/bash

echo "$1"
if [ "$1" == "brew" ]; then
    OPT=''
elif [ "$1" == "apt-get" ]; then
    OPT='-y'
fi

$1 update
$1 install wget $OPT
$1 install tmux $OPT
$1 install vim $OPT
$1 install less $OPT
$1 install at $OPT
$1 install htop $OPT
$1 install bc $OPT

wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O scripts/install_conda.sh
bash scripts/install_conda.sh