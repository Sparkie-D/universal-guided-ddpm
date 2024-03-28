source activate diffusion

ncols=all

for id in 3050
do
    for f in 1 3 5
    do
        for b in 2 10
        do
            for r in 1 10 20 40
            do
                logname=samples/column#${ncols}_id#${id}_f#${f}_b#${b}_r#${r}
                python sample.py --log_name ${logname} --batch_size 256 --n_samples 256 -f ${f} -b ${b} -r ${r} --id ${id}
                python evaluator/evaluate.py  --log_name ${logname} --n_cols ${ncols} --id ${id}
                python evaluator/numeric.py --log_name ${logname} --id ${id}
            done
        done
    done
done