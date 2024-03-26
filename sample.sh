id=010
for f in 1 5 10 20 50 100 200 500
do
    for b in 0 2 5 
    do
        for r in 2 3 4 5 6 7 8 9 10
        do
            logname=samples/sample${id}_f#${f}_b#${b}_r#${r}
            python sample.py --log_name ${logname} --batch_size 256 --n_samples 256 
            python evaluator/evaluate.py  --log_name ${logname} --id ${id}
        done
    done
done