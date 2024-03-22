source activate diffusion

tp=../Datasets/tmp3.csv
vp=../Datasets/diffusion_data/synther2col/pretrain.csv
ep=10000

python train_diffuser.py --train_path ${tp} --valid_path ${vp} --num_epoch ${ep} --log_name 2col_pretrain

fewshot=../Datasets/diffusion_data/synther2col/synther3050/fewshot.csv