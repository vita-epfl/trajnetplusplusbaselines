interaction="directional" 
modes=3
alpha=1.0

#Train
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.vae.trainer --type $interaction --augment --save_every 20 --seed $seed --output alpha${alpha}_seed${seed} --k $modes --alpha_kld $alpha
done

#Get predictions (and submit on AICrowd)
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.vae.trajnet_evaluator --output OUTPUT_BLOCK/trajdata/vae_${interaction}_alpha${alpha}_seed${seed}.pkl --write_only --modes 3
done