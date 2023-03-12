python train.py --n-envs 10 --log-path logs/exp_1  --seed 42
mv logs/exp_1/models/best_model.zip best_model.zip
tar -cvzf submission.tar.gz *