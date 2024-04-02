source activate diffusion

# train diffuser
# logname=california # reg
# cat_cols=

# logname=abalone
# cat_cols='y cat_0'

logname=buddy
cat_cols='y cat_0 cat_1 cat_2 cat_3 cat_4'

# logname=adult
# cat_cols='y cat_0 cat_1 cat_2 cat_3 cat_4 cat_5 cat_6 cat_7'

# logname='king' # reg
# cat_cols='cat_0 cat_1 cat_2'

# tp=../Datasets/diffusion/tabddpm/${logname}/pretrain.csv
# vp=../Datasets/diffusion/tabddpm/${logname}/pretrain.csv
# ep=3000
# python train_diffuser.py --train_path ${tp} --valid_path ${vp} --num_epoch ${ep} --log_name pretrain/${logname} --cat_cols ${cat_cols}

# train discriminator
fewshot=../Datasets/diffusion/tabddpm/${logname}/fewshot.csv
pretrain_path=logs/pretrain/${logname}
python train_discriminator.py --train_path ${fewshot} --valid_path ${fewshot} --log_name fewshot/${logname} --save_model_epoch 100 --num_epoch 1000 --pretrain_path ${pretrain_path} --cat_cols ${cat_cols}
