for dataset in emotion_pairs
do
    for sample_size in 64
    do
        for train_split in 0 1 2
        do
            python -m src.pl_train -c t03b.json+ia3.json+${dataset}.json \
            -k load_weight="t-few/pretrained_checkpoints/t03b_ia3_finish.pt" \
            exp_name=t03b_pretrained/${dataset}/train-${sample_size}-${train_split}/seed${seed} \
            train_split=${train_split} \
            few_shot_random_seed=${seed} \
            seed=${seed} \
            num_shot=$sample_size \
            batch_size=8 \
            eval_batch_size=16 \
            grad_accum_factor=1 \
            eval_before_training=0
        done
    done
done
