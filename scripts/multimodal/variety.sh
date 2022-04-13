interaction="directional" 
additional="--d_steps 0"
modes=3

#Train
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.sgan.trainer --type $interaction --augment --save_every 20 --seed $seed --output seed${seed}_variety --k $modes $additional
done

#Get predictions (and submit on AICrowd)
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.sgan.trajnet_evaluator --output OUTPUT_BLOCK/trajdata/sgan_${interaction}_seed${seed}_variety.pkl --write_only --modes 3
done

