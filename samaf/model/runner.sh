#!/bin/bash
mkdir -p logs
echo "Program started" >> logs/info
date >> logs/info 
python trainer.py
echo "Program exit code: $?" >> logs/info
echo "Program end" >> logs/info
date >> logs/info
sudo shutdown -P now
# watch -n 0.5 nvidia-smi