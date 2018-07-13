import argparse
import time
import random

import pysparkling
import torch

from .. import augmentation
from .. import readers
from .loss import PredictionLoss
from .lstm import LSTM, LSTMPredictor, scene_to_xy
from .pooling import Pooling


class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None):
        self.model = model if model is not None else LSTM()
        self.criterion = criterion if criterion is not None else PredictionLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 10))

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def train(self, scenes, val_scenes, epochs=35):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            preprocess_time = 0.0
            optim_time = 0.0

            print('epoch', epoch)
            self.lr_scheduler.step()

            random.shuffle(scenes)
            epoch_loss = 0.0
            self.model.train()
            for scene_i, scene in enumerate(scenes):
                scene_start = time.time()
                scene = augmentation.random_rotation(scene)
                xy = scene_to_xy(scene).to(self.device)
                preprocess_time += time.time() - scene_start

                optim_start = time.time()
                loss = self.train_batch(xy)
                optim_time += time.time() - optim_start

                epoch_loss += loss

                if scene_i % 100 == 0:
                    print({
                        'type': 'train',
                        'epoch': epoch,
                        'batch': scene_i,
                        'n_batch': len(scenes),
                        'loss': loss,
                    })

            val_loss = 0.0
            eval_start = time.time()
            self.model.train()  # so that it does not return positions but still normals
            for scene in val_scenes:
                xy = scene_to_xy(scene).to(self.device)
                val_loss += self.val_batch(xy)
            eval_time = time.time() - eval_start

            print({
                'train_loss': epoch_loss / len(scenes),
                'val_loss': val_loss / len(val_scenes),
                'duration': time.time() - start_time,
                'preprocess_time': preprocess_time,
                'optim_time': optim_time,
                'eval_time': eval_time,
            })

    def train_batch(self, xy):
        # augmentation: random coordinate shifts
        # noise = (torch.rand_like(xy) - 0.5) * 2.0 * 0.03
        # xy += noise

        # augmentation: random stretching
        # x_scale = 0.9 + 0.2 * random.random()
        # y_scale = 0.9 + 0.2 * random.random()
        # xy[:, :, 0] *= x_scale
        # xy[:, :, 1] *= y_scale

        observed = xy[:9]
        prediction_truth = xy[9:-1].clone()  ## CLONE
        targets = xy[2:, 0] - xy[1:-1, 0]

        # augmentation: vary the length of the observed data a bit
        # truncate_n = int(random.random() * 4.0)
        # observed = observed[truncate_n:]
        # targets = targets[truncate_n:]

        self.optimizer.zero_grad()
        outputs = self.model(observed, prediction_truth)

        loss = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def val_batch(self, xy):
        observed = xy[:9]
        prediction_truth = xy[9:-1].clone()  ## CLONE

        with torch.no_grad():
            outputs = self.model(observed, prediction_truth)

            targets = xy[2:, 0] - xy[1:-1, 0]
            loss = self.criterion(outputs, targets)

        return loss.item()


def main(epochs=35):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs,
                        help='number of epochs')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social'),
                        help='type of LSTM to train')
    parser.add_argument('--train-input-files', default='output/train/**/*.txt',
                        help='glob expression for train input files')
    parser.add_argument('--val-input-files', default='output/val/**/*.txt',
                        help='glob expression for validation input files')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    # configure pooling layer
    pool = None
    if args.type != 'vanilla':
        pool = Pooling(type_=args.type)

    # set model output file
    if args.output is None:
        args.output = 'output/' + args.type + '_lstm.pkl'

    # read in datasets
    sc = pysparkling.Context()
    scenes = (sc
              .wholeTextFiles(args.train_input_files)
              .values()
              .map(readers.trajnet)
              .collect())
    val_scenes = (sc
                  .wholeTextFiles(args.val_input_files)
                  .values()
                  .map(readers.trajnet)
                  .collect())

    model = LSTM(pool=pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = Trainer(model, optimizer=optimizer, device=args.device)
    trainer.train(scenes, val_scenes, epochs=args.epochs)
    LSTMPredictor(trainer.model).save(args.output)


if __name__ == '__main__':
    main()
