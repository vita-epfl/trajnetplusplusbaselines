"""Script to profile trianing.

Run with:
python tests/profile_train.py \
   --train-input-files data/train/biwi_eth/0.txt \
   --val-input-files data/val/biwi_eth/191.txt \
   --type social
"""

import torch
import trajnetbaselines.lstm.trainer
import trajnettools


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    scenes = list(trajnettools.Reader('data/train/biwi_hotel.ndjson').scenes(limit=1))

    pool = trajnetbaselines.lstm.Pooling(type_='social')
    model = trajnetbaselines.lstm.LSTM(pool=pool)
    trainer = trajnetbaselines.lstm.trainer.Trainer(model, device=device)
    with torch.autograd.profiler.profile() as prof:
        trainer.train(scenes, epoch=0)
    prof.export_chrome_trace('profile_trace.json')


if __name__ == '__main__':
    main()
