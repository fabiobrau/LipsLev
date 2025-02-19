cd ../

python3 train.py --datase ag_news --kernel_size 10 --n_classes 4 --lr 100 --seed 1 --valid_size 0 --p 2
python3 train.py --datase ag_news --kernel_size 10 --n_classes 4 --lr 100 --seed 1 --valid_size 0 --p 1
python3 train.py --datase ag_news --kernel_size 10 --n_classes 4 --lr 100 --seed 1 --valid_size 0 --p inf