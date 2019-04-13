#!/bin/bash
source activate deep-learning && conda install -y pytorch torchvision -c pytorch
source activate deep-learning && pip install -r /tmp/requirements.txt

