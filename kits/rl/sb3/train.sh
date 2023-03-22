conda activate luxai_s2
python train.py --n-envs 10 --log-path logs/exp_1  --seed 42
tar -cvzf submission.tar.gz *