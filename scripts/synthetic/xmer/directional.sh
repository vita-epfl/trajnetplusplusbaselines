interaction="directional" 
data_path=synth_456_huge_extra
epochs=15
step_size=6
for seed in 42
do  
    python -m trajnetbaselines.xmer.trainer --path $data_path --goals --type $interaction --augment --epochs $epochs --step_size $step_size --seed $seed --loss 'L2' --output seed${seed}_sequential_tf_extra --batch_size 8 --save_every 1
done
