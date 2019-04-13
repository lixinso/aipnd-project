#!/bin/bash
source activate deep-learning
cd /myworkdir
jupyter notebook --no-browser --port 8888 --ip 0.0.0.0 --allow-root
