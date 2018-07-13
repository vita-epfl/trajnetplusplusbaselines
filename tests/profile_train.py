"""Script to profile trianing.

Run with:
python tests/profile_train.py \
   --train-input-files output/train/biwi_eth/0.txt \
   --val-input-files output/val/biwi_eth/191.txt \
   --type social
"""

import torch
import trajnettools.lstm.trainer


if __name__ == '__main__':
    with torch.autograd.profiler.profile() as prof:
        trajnettools.lstm.trainer.main(epochs=1)
    prof.export_chrome_trace('profile_trace.json')
