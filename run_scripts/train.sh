source activate diffusion

ncols=all
# train diffuser
tp=../Datasets/diffusion/pksim_${ncols}/pretrain.csv
vp=../Datasets/diffusion/pksim_${ncols}/pretrain.csv
ep=10
python train_diffuser.py --train_path ${tp} --valid_path ${vp} --num_epoch ${ep} --log_name pretrain/column#${ncols} --save_model_epoch 5

# train discriminator
num=3050
fewshot=../Datasets/diffusion/pksim_${ncols}/${num}/fewshot.csv
python train_discriminator.py --train_path ${fewshot} --valid_path ${fewshot} --log_name fewshot/column#${ncols}_id#${num} --num_epoch 1000
