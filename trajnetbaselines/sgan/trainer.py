"""Command line tool to train an SGAN model."""

import argparse
import logging
import socket
import sys
import time
import random
import os
import pickle

import numpy as np

import torch
import trajnetplusplustools

from .. import augmentation
from ..lstm.loss import PredictionLoss, L2Loss
from ..lstm.loss import gan_d_loss, gan_g_loss # variety_loss
from ..lstm.gridbased_pooling import GridBasedPooling
from ..lstm.non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from ..lstm.non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention_fast
from ..lstm.more_non_gridbased_pooling import NMMP
from .sgan import SGAN, drop_distant, SGANPredictor
from .sgan import LSTMGenerator, LSTMDiscriminator
from .. import __version__ as VERSION

from ..lstm.utils import center_scene, random_rotation


class Trainer(object):
    def __init__(self, model=None, criterion='L2', optimizer=None, lr_scheduler=None,
                 device=None, batch_size=32, obs_length=9, pred_length=12, augment=False,
                 normalize_scene=False, save_every=1, start_length=0, obs_dropout=False):
        self.model = model if model is not None else SGAN()
        if criterion == 'L2':
            self.criterion = L2Loss(keep_batch_dim=True)
            self.loss_multiplier = 100
        else:
            self.criterion = PredictionLoss(keep_batch_dim=True)
            self.loss_multiplier = 1
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=3e-4, momentum=0.9) # , weight_decay=1e-4
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
                SGANPredictor(self.model).save(state, out + '.epoch{}'.format(epoch))
            self.train(train_scenes, train_goals, epoch)
            self.val(val_scenes, val_goals, epoch)

        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        SGANPredictor(self.model).save(state, out + '.epoch{}'.format(epoch + 1))
        SGANPredictor(self.model).save(state, out)

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

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        d_steps_left = self.model.d_steps
        g_steps_left = self.model.g_steps
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
            
            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                preprocess_time = time.time() - scene_start

                # Decide whether to use the batch for stepping on discriminator or
                # generator; an iteration consists of args.d_steps steps on the
                # discriminator followed by args.g_steps steps on the generator.
                if d_steps_left > 0:
                    step_type = 'd'
                    d_steps_left -= 1
                elif g_steps_left > 0:
                    step_type = 'g'
                    g_steps_left -= 1
                
                ## Train Batch
                loss = self.train_batch(batch_scene, batch_scene_goal, batch_split, step_type)
                epoch_loss += loss
                total_time = time.time() - scene_start

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

                ## Update d_steps, g_steps once they end
                if d_steps_left == 0 and g_steps_left == 0:
                    d_steps_left = self.model.d_steps
                    g_steps_left = self.model.g_steps

            if (scene_i + 1) % (10*self.batch_size) == 0:
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
        self.model.train()  # so that it does not return positions but still normals

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            # make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                # scene_goal = np.array([goals[path[0].pedestrian] for path in paths])
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()
                
                loss_val_batch, loss_test_batch = self.val_batch(batch_scene, batch_scene_goal, batch_split)
                val_loss += loss_val_batch
                test_loss += loss_test_batch

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / (len(scenes)), 3),
            'test_loss': round(test_loss / len(scenes), 3),
            'time': round(eval_time, 1),
        })

    def train_batch(self, batch_scene, batch_scene_goal, batch_split, step_type):
        """Training of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene
        step_type : String ('g', 'd')
            Determines whether to train generator or discriminator

        Returns
        -------
        loss : scalar
            Training loss of the batch
        """

        observed = batch_scene[self.start_length:self.obs_length].clone()
        prediction_truth = batch_scene[self.obs_length:].clone()
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]

        rel_output_list, outputs, scores_real, scores_fake = self.model(observed, batch_scene_goal, batch_split, prediction_truth, step_type=step_type)

        loss = self.loss_criterion(rel_output_list, targets, batch_split, scores_fake, scores_real, step_type)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_batch(self, batch_scene, batch_scene_goal, batch_split):
        """Validation of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided
        """

        observed = batch_scene[self.start_length:self.obs_length]
        prediction_truth = batch_scene[self.obs_length:].clone()  ## CLONE
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]
        
        with torch.no_grad():
            rel_output_list, _, _, _ = self.model(observed, batch_scene_goal, batch_split, n_predict=self.pred_length)

            ## top-k loss
            loss = self.variety_loss(rel_output_list, targets, batch_split)

        return 0.0, loss.item()

    def loss_criterion(self, rel_output_list, targets, batch_split, scores_fake, scores_real, step_type):
        """ Loss calculation function

        Parameters
        ----------
        rel_output_list : List of length k
            Each element of the list is Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        targets : Tensor [pred_length, batch_size, 2]
            Groundtruth sequence of primary pedestrians of each scene
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the primary tracks of each scene
        scores_real : Tensor [batch_size, ]
            Discriminator scores of groundtruth primary tracks
        scores_fake : Tensor [batch_size, ]
            Discriminator scores of prediction primary tracks
        step_type : 'g' / 'd'
            Determines whether to train the generator / discriminator

        Returns
        -------
        loss : Tensor [1,]
            The corresponding generator / discriminator loss
        """

        if step_type == 'd':
            loss = gan_d_loss(scores_real, scores_fake)

        else:
            ## top-k loss
            loss = self.variety_loss(rel_output_list, targets, batch_split)

            ## If discriminator used.
            if self.model.use_d:
                loss += gan_g_loss(scores_fake)

        return loss

    def variety_loss(self, inputs, target, batch_split):
        """ Variety loss calculation as proposed in SGAN

        Parameters
        ----------
        inputs : List of length k
            Each element of the list is Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        target : Tensor [pred_length, num_tracks, 2]
            Groundtruth sequence of primary pedestrians of each scene
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the primary tracks of each scene

        Returns
        -------
        loss : Tensor [1,]
            variety loss
        """

        iterative_loss = [] 
        for sample in inputs:
            sample_loss = self.criterion(sample[-self.pred_length:], target, batch_split) * self.loss_multiplier
            iterative_loss.append(sample_loss)

        loss = torch.stack(iterative_loss)
        loss = torch.min(loss, dim=0)[0]
        loss = torch.sum(loss)
        return loss

def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals 
    
    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    """

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
            goal_dict = pickle.load(open('goal_files/' + subset + file +'.pkl', "rb"))
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
                        help='step_size of lr scheduler')
    parser.add_argument('--save_every', default=1, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp'),
                        help='type of interaction encoder')
    parser.add_argument('--norm_pool', action='store_true',
                        help='normalize the scene along direction of movement')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling (only consider pedestrian in front along direction of movement)')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='augment scenes (rotation augmentation)')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of oservation')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goal_path', default=None,
                        help='glob expression for goal files')
    parser.add_argument('--loss', default='L2', choices=('L2', 'pred'),
                        help='loss objective to train the model')
    parser.add_argument('--goals', action='store_true',
                        help='flag to use goals')

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
                                 help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=16,
                                 help='number of cells per side')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512],
                                 help='interaction module layer dims (for gridbased pooling)')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='output dimension of pooling/interaction vector')
    hyperparameters.add_argument('--embedding_arch', default='two_layer',
                                 help='interaction encoding arch for gridbased pooling')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal dimension')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='attentionmlp spatial dimension')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='attentionmlp vel dimension')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background value of gridbased pooling')
    hyperparameters.add_argument('--sample', default=1.0, type=float,
                                 help='sample ratio of train/val scenes')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for grid-based')
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='flag to not consider velocity in nn')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='number of neighbours to consider in DirectConcat')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iters in NMMP')
    hyperparameters.add_argument('--obs_dropout', action='store_true',
                                 help='obs length dropout (regularization)')
    hyperparameters.add_argument('--start_length', default=0, type=int,
                                 help='start length during obs dropout')

    ## SGAN-Specific
    hyperparameters.add_argument('--k', type=int, default=3,
                                 help='number of samples for variety loss')
    hyperparameters.add_argument('--noise_dim', type=int, default=16,
                                 help='dimension of z')
    hyperparameters.add_argument('--add_noise', action='store_true',
                                 help='To Add Noise')
    hyperparameters.add_argument('--noise_type', default='gaussian',
                                 help='type of noise to be added')
    hyperparameters.add_argument('--discriminator', action='store_true',
                                 help='discriminator to be added')
    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.goals:
        args.output = 'OUTPUT_BLOCK/{}/sgan_goals_{}_{}.pkl'.format(args.path, args.type, args.output)
    else:
        args.output = 'OUTPUT_BLOCK/{}/sgan_{}_{}.pkl'.format(args.path, args.type, args.output)

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

    # create interaction/pooling modules
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
    elif args.type == 's_att_fast':
        pool = SAttention_fast(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type != 'vanilla':
        pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
                                cell_side=args.cell_side, n=args.n, front=args.front,
                                out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
                                constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
                                norm=args.norm, layer_dims=args.layer_dims)

    # generator
    lstm_generator = LSTMGenerator(embedding_dim=args.coordinate_embedding_dim, hidden_dim=args.hidden_dim,
                                   pool=pool, goal_flag=args.goals, goal_dim=args.goal_dim, noise_dim=args.noise_dim,
                                   add_noise=args.add_noise, noise_type=args.noise_type)

    # discriminator
    print("discriminator: ", args.discriminator)
    lstm_discriminator = None
    if args.discriminator:
        lstm_discriminator = LSTMDiscriminator(embedding_dim=args.coordinate_embedding_dim,
                                               hidden_dim=args.hidden_dim, pool=pool,
                                               goal_flag=args.goals, goal_dim=args.goal_dim)

    # GAN model
    model = SGAN(generator=lstm_generator, discriminator=lstm_discriminator,
                 add_noise=args.add_noise, k=args.k)

    # optimizer and schedular
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

        if args.load_full_state:
        # load optimizers from last training
        # useful to continue model training
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
