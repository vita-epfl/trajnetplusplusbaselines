interaction="social" 
additional="--n 16 --embedding_arch two_layer --layer_dims 1024" 

#Train
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.lstm.trainer --type $interaction $additional --augment --save_every 20 --seed $seed --output seed${seed}
done

#Get predictions (and submit on AICrowd)
for seed in 42 10 20 30 40
do  
    python -m trajnetbaselines.lstm.trajnet_evaluator --output OUTPUT_BLOCK/trajdata/lstm_${interaction}_seed${seed}.pkl --write_only
done