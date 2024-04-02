source activate diffusion

# name=abalone
# cat_cols='y cat_0'

# name=adult  # classification
# cat_cols='y cat_0 cat_1 cat_2 cat_3 cat_4 cat_5 cat_6 cat_7'

name=buddy
cat_cols='y cat_0 cat_1 cat_2 cat_3 cat_4'

# name=california # regression
# cat_cols=

# name='king' # regression
# cat_cols='cat_0 cat_1 cat_2'


pretrain_path=logs/pretrain/${name}
fewshot_path=logs/fewshot/${name}
data_dir=../Datasets/diffusion/tabddpm/${name}
for f in 2 # 2 3 3 5
do
    for b in 0 # 2 3 5 10
    do
        for r in 2 # 2 3 # 3 5 7 10 20 40
        do
            logname=samples/${name}/f#${f}_b#${b}_r#${r}
            python sample.py --log_name ${logname} --batch_size 256 --pretrain_path ${pretrain_path} --fewshot_path ${fewshot_path} --n_samples 256 -f ${f} -b ${b} -r ${r} 
            python evaluator/distribution.py --log_name ${logname} --pretrain_path ${pretrain_path} --data_path ${data_dir} --cat_cols ${cat_cols}
            python evaluator/numeric.py --log_name ${logname} --pretrain_path ${pretrain_path} --data_path ${data_dir} --cat_cols ${cat_cols}
            python evaluator/mle.py --log_name ${logname} --pretrain_path ${pretrain_path} --data_path ${data_dir} --task_type clf
        done
    done
done