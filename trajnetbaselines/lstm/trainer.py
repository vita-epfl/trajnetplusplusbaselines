import argparse
import datetime
import logging
import time
import random

import torch
import trajnettools

from .. import augmentation
from .loss import PredictionLoss
from .lstm import LSTM, LSTMPredictor, drop_distant
from .pooling import Pooling
from .. import __version__ as VERSION


class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None):
        self.model = model if model is not None else LSTM()
        self.criterion = criterion if criterion is not None else PredictionLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 15))

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)

    def loop(self, train_scenes, val_scenes, out, epochs=35):
        for epoch in range(epochs):
            LSTMPredictor(self.model).save(out + '.epoch{}'.format(epoch))
            self.train(train_scenes, epoch)
            self.val(val_scenes, epoch)
        LSTMPredictor(self.model).save(out)

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, scenes, epoch):
        start_time = time.time()

        print('epoch', epoch)
        self.lr_scheduler.step()

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        for scene_i, (_, scene) in enumerate(scenes):
            scene_start = time.time()
            scene = drop_distant(scene)
            scene = augmentation.random_rotation(scene)
            scene = torch.Tensor(scene).to(self.device)
            preprocess_time = time.time() - scene_start

            loss = self.train_batch(scene)
            epoch_loss += loss
            total_time = time.time() - scene_start

            if scene_i % 10 == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.lr(),
                    'loss': round(loss, 3),
                })

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / len(scenes), 3),
            'time': round(time.time() - start_time, 1),
        })

    def val(self, val_scenes, epoch):
        val_loss = 0.0
        eval_start = time.time()
        self.model.train()  # so that it does not return positions but still normals
        for _, scene in val_scenes:
            scene = drop_distant(scene)
            scene = torch.Tensor(scene).to(self.device)
            val_loss += self.val_batch(scene)
        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / len(val_scenes), 3),
            'time': round(eval_time, 1),
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
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social'),
                        help='type of LSTM to train')
    parser.add_argument('--train-input-files', default='data/train/**/*.ndjson',
                        help='glob expression for train input files')
    parser.add_argument('--val-input-files', default='data/val/**/*.ndjson',
                        help='glob expression for validation input files')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')

    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--pre-epochs', default=5, type=int,
                          help='number of epochs')
    pretrain.add_argument('--pre-lr', default=1e-3, type=float,
                          help='initial learning rate')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='RNN hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')

    args = parser.parse_args()

    # set model output file
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    if args.output is None:
        args.output = 'output/{}_lstm_{}.pkl'.format(args.type, timestamp)

    # configure logging
    from pythonjsonlogger import jsonlogger
    import socket
    import sys
    file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    # read in datasets
    train_scenes = list(trajnettools.load_all(args.train_input_files,
                                              sample={'syi.ndjson': 0.05}))
    val_scenes = list(trajnettools.load_all(args.val_input_files,
                                            sample={'syi.ndjson': 0.05}))

    # create model
    pool = None
    if args.type != 'vanilla':
        pool = Pooling(type_=args.type, hidden_dim=args.hidden_dim)
    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim)

    # train
    if args.load_state:
        with open(args.load_state, 'rb') as f:
            pretrained_state_dict = torch.load(f)
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        # pre-optimize all parameters that were not in the loaded state dict
        pretrained_params = set(pretrained_state_dict.keys())
        untrained_params = [p for n, p in model.named_parameters()
                            if n not in pretrained_params]
        pre_optimizer = torch.optim.Adam(untrained_params, lr=args.pre_lr, weight_decay=1e-4)
        pre_trainer = Trainer(model, optimizer=pre_optimizer, device=args.device)
        for pre_epoch in range(-args.pre_epochs, 0):
            pre_trainer.train(train_scenes, epoch=pre_epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = Trainer(model, optimizer=optimizer, device=args.device)
    trainer.loop(train_scenes, val_scenes, args.output, epochs=args.epochs)


if __name__ == '__main__':
    main()
