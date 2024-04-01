source activate diffusion

name=insurance
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
            python evaluator/distribution.py  --log_name ${logname} --pretrain_path ${pretrain_path} --data_path ${data_dir}
            python evaluator/numeric.py       --log_name ${logname} --pretrain_path ${pretrain_path} --data_path ${data_dir}
        done
    done
done