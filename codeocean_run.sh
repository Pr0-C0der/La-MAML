apt-get update

apt-get install wget

apt-get install python3-pip # terminal

conda create -n env python=3.9.1 --yes

conda activate env

pip install --upgrade wandb

pip install ipdb

conda install matplotlib numpy pillow urllib3 scipy --yes # terminal

conda update -n base -c defaults conda --yes #terminal

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch --yes # terminal

# cd La-MAML

# git pull https://github.com/joeljosephjin/La-MAML

wandb login 665a5d573c302c27f7dab355484da17a460e6759

# chmod +x run_many.sh
# ./run_many.sh

CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
SEED=0

##### ER ##### CIFAR DATASET Multi-Pass 
python3 mainwb.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 1 --n_epochs 10 \
                     --lr 0.03 --glances 1 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1