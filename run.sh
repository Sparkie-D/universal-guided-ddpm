source activate diffusion

tp=../Datasets/diffusion_data/synther2col/pretrain.csv
vp=../Datasets/diffusion_data/synther2col/pretrain.csv
ep=100000

# python train_diffuser.py --train_path ${tp} --valid_path ${vp} --num_epoch ${ep} --log_name 2col_pretrain_128step

num=010
fewshot=../Datasets/diffusion_data/synther2col/synther${num}/fewshot.csv

python train_disc.py --train_path ${fewshot} --valid_path ${fewshot} --num_epoch ${ep} --log_name fewshot${num}_sigmoid --num_epoch 30000