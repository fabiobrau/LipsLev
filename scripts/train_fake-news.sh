cd ../

python3 train.py --datase fake-news --kernel_size 10 --n_classes 2 --lr 100 --seed 1 --valid_size 0 --p 2
python3 train.py --datase fake-news --kernel_size 10 --n_classes 2 --lr 100 --seed 1 --valid_size 0 --p 1
python3 train.py --datase fake-news --kernel_size 10 --n_classes 2 --lr 100 --seed 1 --valid_size 0 --p inf