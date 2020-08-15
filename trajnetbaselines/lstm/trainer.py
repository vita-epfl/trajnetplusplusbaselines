"""Command line tool to train an LSTM model."""

import argparse
import logging
import socket
import sys
import time
import random
import os
import pickle
import torch
import numpy as np

import trajnetplusplustools

from .. import augmentation
from .loss import PredictionLoss, L2Loss
from .lstm import LSTM, LSTMPredictor, drop_distant
from .gridbased_pooling import GridBasedPooling
from .non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from .non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention, SAttention_fast
from .more_non_gridbased_pooling import NMMP

from .. import __version__ as VERSION

from .utils import center_scene, random_rotation

class Trainer(object):
    def __init__(self, model=None, criterion='L2', optimizer=None, lr_scheduler=None,
                 device=None, batch_size=32, obs_length=9, pred_length=12, augment=False,
                 normalize_scene=False, save_every=1, start_length=0, obs_dropout=False):
        self.model = model if model is not None else LSTM()
        if criterion == 'L2':
            self.criterion = L2Loss()
            self.loss_multiplier = 100
        else:
            self.criterion = PredictionLoss()
            self.loss_multiplier = 1
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=3e-4, momentum=0.9)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 15))

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        self.batch_size = batch_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length+self.pred_length

        self.augment = augment
        self.normalize_scene = normalize_scene

        self.start_length = start_length
        self.obs_dropout = obs_dropout

    def loop(self, train_scenes, val_scenes, train_goals, val_goals, out, epochs=35, start_epoch=0):
        for epoch in range(start_epoch, start_epoch + epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.lr_scheduler.state_dict()}
                LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch))
            self.train(train_scenes, train_goals, epoch)
            self.val(val_scenes, val_goals, epoch)


        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch + 1))
        LSTMPredictor(self.model).save(state, out)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, scenes, goals, epoch):
        start_time = time.time()

        print('epoch', epoch)
        self.lr_scheduler.step()

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)
                # scene = augmentation.add_noise(scene, thresh=0.01)

            scene = torch.Tensor(scene).to(self.device)
            scene_goal = torch.Tensor(scene_goal).to(self.device)
            preprocess_time = time.time() - scene_start

            loss, _ = self.train_batch(scene, scene_goal)
            epoch_loss += loss
            total_time = time.time() - scene_start

            if (scene_i + 1) % self.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (scene_i + 1) % 10 == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.get_lr(),
                    'loss': round(loss, 3),
                })

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 5),
            'time': round(time.time() - start_time, 1),
        })

    def val(self, scenes, goals, epoch):
        eval_start = time.time()

        val_loss = 0.0
        test_loss = 0.0
        self.model.train()
        for _, (filename, scene_id, paths) in enumerate(scenes):
            # make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            scene = torch.Tensor(scene).to(self.device)
            scene_goal = torch.Tensor(scene_goal).to(self.device)
            loss_1, loss_2 = self.val_batch(scene, scene_goal)
            val_loss += loss_1
            test_loss += loss_2
        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / (len(scenes)), 3),
            'test_loss': round(test_loss / len(scenes), 3),
            'time': round(eval_time, 1),
        })

    def train_batch(self, xy, goals):
        ## If observation dropout active
        if self.obs_dropout:
            self.start_length = random.randint(0, self.obs_length - 2)

        observed = xy[self.start_length:self.obs_length].clone()
        prediction_truth = xy[self.obs_length:self.seq_length-1].clone()  ## CLONE
        targets = xy[self.obs_length:self.seq_length] - xy[self.obs_length-1:self.seq_length-1]

        rel_outputs, outputs = self.model(observed, goals, prediction_truth)

        ## Loss wrt primary only
        nan_list = torch.isnan(xy[self.obs_length:self.seq_length]).view(-1)
        loss = self.criterion(rel_outputs[-self.pred_length:], targets, nan_list) * self.loss_multiplier

        loss.backward()

        return loss.item(), outputs.detach()

    def val_batch(self, xy, goals):
        if self.obs_dropout:
            self.start_length = 0
        observed = xy[self.start_length:self.obs_length]
        prediction_truth = xy[self.obs_length:self.seq_length-1].clone()  ## CLONE
        observed_test = observed.clone()

        with torch.no_grad():
            ## GT of neighbours given (Better validation curve to monitor model)
            rel_outputs, _ = self.model(observed, goals, prediction_truth)
            nan_list = torch.isnan(xy[self.obs_length:self.seq_length]).view(-1)
            targets = xy[self.obs_length:self.seq_length] - xy[self.obs_length-1:self.seq_length-1]
            loss = self.criterion(rel_outputs[-self.pred_length:], targets, nan_list) * self.loss_multiplier

            ## GT of neighbours not Given
            rel_outputs_test, _ = self.model(observed_test, goals, n_predict=self.pred_length)
            loss_test = self.criterion(rel_outputs_test[-self.pred_length:], targets, nan_list) * self.loss_multiplier

        return loss.item(), loss_test.item()

def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals """

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('dest_new/' + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None

def main(epochs=50):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--step_size', default=15, type=int,
                        help='step_size of scheduler')
    parser.add_argument('--save_every', default=1, type=int,
                        help='frequency of saving model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 's_att', 'nn_tag',
                                 'nmmp'),
                        help='type of LSTM to train')
    parser.add_argument('--norm_pool', action='store_true',
                        help='normalize_pool (along direction of movement)')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling (only consider pedestrian in front along direction of movement)')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='augment scenes')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goal_path', default=None,
                        help='glob expression for goal files')
    parser.add_argument('--loss', default='L2',
                        help='loss function')
    parser.add_argument('--goals', action='store_true',
                        help='to use goals')

    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ##Pretrain Pooling AE
    pretrain.add_argument('--load_pretrained_pool_path', default=None,
                          help='load a pickled model state dictionary of pool AE before training')
    pretrain.add_argument('--pretrained_pool_arch', default='onelayer',
                          help='architecture of pool representation')
    pretrain.add_argument('--downscale', type=int, default=4,
                          help='downscale factor of pooling grid')
    pretrain.add_argument('--finetune', type=int, default=0,
                          help='finetune factor of pretrained model')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='RNN hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=16,
                                 help='number of cells per side')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*',
                                 help='interaction module layer dims for gridbased pooling')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='pooling dimension')
    hyperparameters.add_argument('--embedding_arch', default='two_layer',
                                 help='interaction arch')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal dimension')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='attention mlp spatial dimension')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='attention mlp vel dimension')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background of pooling grid')
    hyperparameters.add_argument('--sample', default=1.0, type=float,
                                 help='sample ratio of train/val scenes')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for grid-based')
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='dont consider velocity in nn')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='neighbours to consider in DirectConcat')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iters in NMMP')
    hyperparameters.add_argument('--start_length', default=0, type=int,
                                 help='prediction length')
    hyperparameters.add_argument('--obs_dropout', action='store_true',
                                 help='obs length dropout')
    args = parser.parse_args()

    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.goals:
        args.output = 'OUTPUT_BLOCK/{}/lstm_goals_{}_{}.pkl'.format(args.path, args.type, args.output)
    else:
        args.output = 'OUTPUT_BLOCK/{}/lstm_{}_{}.pkl'.format(args.path, args.type, args.output)

    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
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
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    args.path = 'DATA_BLOCK/' + args.path
    ## Prepare data
    train_scenes, train_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)
    val_scenes, val_goals = prepare_data(args.path, subset='/val/', sample=args.sample, goals=args.goals)

    ## pretrained pool model (if any)
    pretrained_pool = None

    # create model (Various interaction/pooling modules)
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                     mlp_dim_vel=args.vel_dim)
    elif args.type == 'nmmp':
        pool = NMMP(hidden_dim=args.hidden_dim, out_dim=args.pool_dim, k=args.mp_iters)
    elif args.type == 'attentionmlp':
        pool = AttentionMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                   mlp_dim_spatial=args.spatial_dim, mlp_dim_vel=args.vel_dim)
    elif args.type == 'directionalmlp':
        pool = DirectionalMLPPooling(out_dim=args.pool_dim)
    elif args.type == 'nn':
        pool = NN_Pooling(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
    elif args.type == 'nn_lstm':
        pool = NN_LSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 'traj_pool':
        pool = TrajectronPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 's_att':
        pool = SAttention(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 's_att_fast':
        pool = SAttention_fast(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type != 'vanilla':
        pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
                                cell_side=args.cell_side, n=args.n, front=args.front,
                                out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
                                constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
                                norm=args.norm, layer_dims=args.layer_dims)

    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim,
                 goal_flag=args.goals,
                 goal_dim=args.goal_dim)

    # optimizer
    if args.finetune == 0:
        print("NO Finetune")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        print("Finetune") ## Can Finetune other components as well
        params = list([param for name, param in model.named_parameters() if 'pool' in name]) ##pretrained_model
        base_params = list([param for name, param in model.named_parameters() if 'pool' not in name])
        optimizer = torch.optim.Adam([{'params': base_params}, {'params': params, 'lr': (args.lr/args.finetune)}], lr=args.lr, weight_decay=1e-4)

    lr_scheduler = None
    if args.step_size is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
    start_epoch = 0

    # train
    if args.load_state:
        # load pretrained model.
        # useful for tranfer learning
        print("Loading Model Dict")
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        ## Partial Dict Loading
        # partial_pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not (('encoder' in k) or ('decoder' in k))}
        # state = model.state_dict()
        # state.update(partial_pretrained_state_dict)
        # model.load_state_dict(state)

        if args.load_full_state:
        # load optimizers from last training
        # useful to continue training
            print("Loading Optimizer Dict")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # , weight_decay=1e-4
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

    #trainer
    trainer = Trainer(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
                      criterion=args.loss, batch_size=args.batch_size, obs_length=args.obs_length,
                      pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
                      save_every=args.save_every, start_length=args.start_length, obs_dropout=args.obs_dropout)
    trainer.loop(train_scenes, val_scenes, train_goals, val_goals, args.output, epochs=args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
