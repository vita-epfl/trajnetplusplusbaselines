interaction="directional" 
epochs=20

#Train
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.lstm.trainer --type $interaction --augment --save_every 20 --epochs $epochs --seed $seed --output seed${seed}
done

#Get predictions (and submit on AICrowd)
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.lstm.trajnet_evaluator --output OUTPUT_BLOCK/trajdata/lstm_${interaction}_seed${seed}.pkl --write_only
done