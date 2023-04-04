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

# trajnetbaselines.lstm
from trajnetbaselines import augmentation
from trajnetbaselines.lstm.loss import PredictionLoss, L2Loss
# from trajnetbaselines.lstm.lstm import LSTM, LSTMPredictor, drop_distant
from trajnetbaselines.lstm.lstm import LSTMPredictor, drop_distant
from trajnetbaselines.lstm.gridbased_pooling import GridBasedPooling
from trajnetbaselines.lstm.non_gridbased_pooling import NearestNeighborMLP, HiddenStateMLPPooling, AttentionMLPPooling
from trajnetbaselines.lstm.non_gridbased_pooling import NearestNeighborLSTM, TrajectronPooling

from trajnetbaselines import __version__ as VERSION

from trajnetbaselines.lstm.utils import center_scene, random_rotation
from trajnetbaselines.lstm.data_load_utils import prepare_data
# WJF: for ABT
from trajnetbaselines.lstm.process_args import get_train_args
from trajnetbaselines.lstm.models.autobot_joint import AutoBotJoint
from trajnetbaselines.lstm.utilsABT.train_helpers import nll_loss_multimodes, nll_loss_multimodes_joint
from torch.optim.lr_scheduler import MultiStepLR
# END

class Trainer(object):
    # By WJF: for ABT
    def __init__(self, args, device=None, obs_length=9, pred_length=12, save_every=5, start_length=0,val_flag=True):
    # END
        # self.model = model if model is not None else LSTM()
        # self.criterion = criterion if criterion is not None else PredictionLoss()
        # self.optimizer = optimizer if optimizer is not None else \
        #                  torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # self.lr_scheduler = lr_scheduler if lr_scheduler is not None else \
        #                     torch.optim.lr_scheduler.StepLR(self.optimizer, 15)

        self.device = device if device is not None else torch.device('cpu')
        # self.model = self.model.to(self.device)
        # self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length+self.pred_length

        # self.augment = augment
        # self.augment_noise = augment_noise
        # self.normalize_scene = normalize_scene

        self.start_length = start_length
        # self.obs_dropout = obs_dropout

        self.val_flag = val_flag

        # WJF: For ABT
        self.args=args
        self.batch_size = self.args.batch_size
        
        if "real" in self.args.path:
            # self.num_other_agents = 35
            # self.num_other_agents = 31
            # self.num_other_agents = 30
            # self.num_other_agents = 24
            # self.num_other_agents = 14
            # self.num_other_agents = 9
            # self.num_other_agents = 19
            self.num_other_agents = 11
            ##
            # self.num_other_agents = 10
            # self.num_other_agents = 12
            # self.num_other_agents = 13
        elif "synth" in self.args.path:
            self.num_other_agents = 5

        self.pred_horizon = 12
        self.num_agent_types = 1  # code assuming only one type of agent (pedestrians).
        self.in_seq_len = 9
        self.predict_yaw = False
        self.map_attr = 0  # dummy
        self.k_attr=2
        self.initialize_model()

        self.optimiser = torch.optim.Adam(self.autobot_model.parameters(), lr=self.args.learning_rate,
                                    eps=self.args.adam_epsilon)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=args.learning_rate_sched, gamma=0.5,
                                               verbose=True)
        # END

    def initialize_model(self):
        self.autobot_model = AutoBotJoint(k_attr=self.k_attr,
                                            d_k=self.args.hidden_size,
                                            _M=self.num_other_agents,
                                            c=self.args.num_modes,
                                            T=self.pred_horizon,
                                            L_enc=self.args.num_encoder_layers,
                                            dropout=self.args.dropout,
                                            num_heads=self.args.tx_num_heads,
                                            L_dec=self.args.num_decoder_layers,
                                            tx_hidden_size=self.args.tx_hidden_size,
                                            use_map_lanes=self.args.use_map_lanes,
                                            map_attr=self.map_attr,
                                            num_agent_types=self.num_agent_types,
                                            predict_yaw=self.predict_yaw).to(self.device)

    def loop(self, train_scenes, val_scenes, train_goals, val_goals, out, epochs=35, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'state_dict': self.autobot_model.state_dict(),
                         'optimizer': self.optimiser.state_dict(),
                         'scheduler': self.optimiser_scheduler.state_dict()}
                LSTMPredictor(self.autobot_model).save(state, out + '.epoch{}'.format(epoch) + '.N{}'.format(self.num_other_agents+1) + '.Lr{}'.format(self.args.learning_rate) + '.Head{}'.format(self.args.tx_num_heads) + '.E{}'.format(self.args.num_encoder_layers) + '.D{}'.format(self.args.num_decoder_layers) + '.HSize{}'.format(self.args.hidden_size) + 'TXSize{}'.format(self.args.tx_hidden_size) + 'c{}'.format(self.args.num_modes)+ 'Se{}'.format(self.args.seed)+ 'bt{}'.format(self.args.batch_size))
            self.train(train_scenes, train_goals, epoch)
            if self.val_flag:
                self.val(val_scenes, val_goals, epoch)


        state = {'epoch': epoch + 1, 'state_dict': self.autobot_model.state_dict(),
                 'optimizer': self.optimiser.state_dict(),
                 'scheduler': self.optimiser_scheduler.state_dict()}
        LSTMPredictor(self.autobot_model).save(state, out + '.epoch{}'.format(epoch) + '.N{}'.format(self.num_other_agents+1) + '.Lr{}'.format(self.args.learning_rate) + '.Head{}'.format(self.args.tx_num_heads) + '.E{}'.format(self.args.num_encoder_layers) + '.D{}'.format(self.args.num_decoder_layers) + '.HSize{}'.format(self.args.hidden_size) + 'TXSize{}'.format(self.args.tx_hidden_size) + 'c{}'.format(self.args.num_modes) + 'Se{}'.format(self.args.seed)+ 'bt{}'.format(self.args.batch_size))
        LSTMPredictor(self.autobot_model).save(state, out)

    def get_lr(self):
        for param_group in self.optimiser.param_groups:
            return param_group['lr']

    def train(self, scenes, goals, epoch):
        start_time = time.time()

        print('epoch', epoch)
        random.shuffle(scenes)
        epoch_loss = 0.0
        self.autobot_model.train()
        self.optimiser.zero_grad()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        N = self.num_other_agents + 1
        largest_num_agents = 0.0

        ego_in_return = np.zeros((self.batch_size, 9, 3))
        ego_out_return = np.zeros((self.batch_size, 12, 3))
        agents_in_return = np.zeros((self.batch_size, 9, N-1, 3))
        agents_out_return = np.zeros((self.batch_size, 12, N-1, 3))
        roads_return = np.zeros((self.batch_size, 1, 1))
        agent_types_return = np.zeros((self.batch_size, N, 1))

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            curr_scene = drop_distant(scene, max_num_peds=N)
            curr_scene, _, _ = center_scene(curr_scene)

            if curr_scene.shape[1] > largest_num_agents:
                largest_num_agents = curr_scene.shape[1]

            if curr_scene.shape[1] < N:
                # Need to pad array to have shape 21xNx2
                temp_curr_scene = np.zeros((21, N, 2))
                temp_curr_scene[:, :, :] = np.nan
                temp_curr_scene[:, :curr_scene.shape[1], :] = curr_scene
                curr_scene = temp_curr_scene.copy()
            scene = curr_scene

            data_mask = np.ones((scene.shape[0], scene.shape[1], 3))
            data_mask[:, :, :2] = scene
            nan_indices = np.where(np.isnan(scene[:, :, 0]))
            data_mask[nan_indices] = [0, 0, 0]

            # Separate past and future.
            agents_in = data_mask[:9]
            agents_out = data_mask[9:]

            ego_in = agents_in[:, 0]
            ego_out = agents_out[:, 0]

            agent_types = np.ones((N, 1))
            roads = np.ones((1, 1))  # for dataloading to work with other datasets that have images.

            ego_in_return[(scene_i) % self.batch_size] = ego_in
            ego_out_return[(scene_i) % self.batch_size] = ego_out
            agents_in_return[(scene_i) % self.batch_size] = agents_in[:, 1:]
            agents_out_return[(scene_i) % self.batch_size] = agents_out[:, 1:]
            roads_return[(scene_i) % self.batch_size] = roads
            agent_types_return[(scene_i) % self.batch_size] = agent_types

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                ego_in_return = torch.Tensor(ego_in_return).to(self.device)
                ego_out_return = torch.Tensor(ego_out_return).to(self.device)
                agents_in_return = torch.Tensor(agents_in_return).to(self.device)
                agents_out_return = torch.Tensor(agents_out_return).to(self.device)
                roads_return = torch.Tensor(roads_return).to(self.device)
                agent_types_return = torch.Tensor(agent_types_return).to(self.device)

                batch_split = torch.Tensor(batch_split).to(self.device).long()

                preprocess_time = time.time() - scene_start

                ## Train Batch
                loss = self.train_batch(ego_in_return, ego_out_return, agents_in_return, agents_out_return, roads_return, agent_types_return, batch_split, batch_scene)
                epoch_loss += loss
                total_time = time.time() - scene_start

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

                largest_num_agents = 0.0
                ego_in_return = np.zeros((self.batch_size, 9, 3))
                ego_out_return = np.zeros((self.batch_size, 12, 3))
                agents_in_return = np.zeros((self.batch_size, 9, N-1, 3))
                agents_out_return = np.zeros((self.batch_size, 12, N-1, 3))
                roads_return = np.zeros((self.batch_size, 1, 1))
                agent_types_return = np.zeros((self.batch_size, N, 1))

            if (scene_i + 1) % (10*self.batch_size) == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.get_lr(),
                    'loss': round(loss, 3),
                })

        self.optimiser_scheduler.step()
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

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
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

    def train_batch(self, ego_in, ego_out, agents_in, agents_out, roads, agent_types, batch_split, batch_scene):
        
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

        Returns
        -------
        loss : scalar
            Training loss of the batch
        """

        pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads, agent_types)
        
        nll_loss, kl_loss, post_entropy, adefde_loss = \
                    nll_loss_multimodes_joint(pred_obs, ego_out, agents_out, mode_probs,
                                              entropy_weight=self.args.entropy_weight,
                                              kl_weight=self.args.kl_weight,
                                              use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss,
                                              agent_types=agent_types,
                                              predict_yaw=self.predict_yaw)
        
        loss = (nll_loss + adefde_loss + kl_loss)

        self.optimiser.zero_grad()
        (nll_loss + adefde_loss + kl_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
        self.optimiser.step()

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
            is provided
        loss_test : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided (evaluation scenario)
        """

        if self.obs_dropout:
            self.start_length = 0

        observed = batch_scene[self.start_length:self.obs_length]
        prediction_truth = batch_scene[self.obs_length:self.seq_length-1].clone()  ## CLONE
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]
        observed_test = observed.clone()

        with torch.no_grad():
            ## groundtruth of neighbours provided (Better validation curve to monitor model)
            rel_outputs, _ = self.model(observed, batch_scene_goal, batch_split, prediction_truth)
            loss = self.criterion(rel_outputs[-self.pred_length:], targets, batch_split) * self.batch_size

            ## groundtruth of neighbours not provided
            rel_outputs_test, _ = self.model(observed_test, batch_scene_goal, batch_split, n_predict=self.pred_length)
            loss_test = self.criterion(rel_outputs_test[-self.pred_length:], targets, batch_split) * self.batch_size

        return loss.item(), loss_test.item()

def main(epochs=25):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', default=epochs, type=int,
    #                     help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,
                        help='starting time step of encoding observation')
    # parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--lr', default=1e-3, type=float,
    #                     help='initial learning rate')
    # parser.add_argument('--step_size', default=10, type=int,
    #                     help='step_size of lr scheduler')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    # parser.add_argument('--disable-cuda', action='store_true',
    #                     help='disable CUDA')
    # parser.add_argument('--path', default='trajdata',
    #                     help='glob expression for data files')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians')
    # parser.add_argument('--loss', default='pred', choices=('L2', 'pred'),
    #                     help='loss objective, L2 loss (L2) and Gaussian loss (pred)')
    # parser.add_argument('--type', default='vanilla',
    #                     choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp',
    #                              'nn', 'attentionmlp', 'nn_lstm', 'traj_pool'),
    #                     help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    # parser.add_argument('--seed', type=int, default=42)

    ## Augmentations
    # parser.add_argument('--augment', action='store_true',
    #                     help='perform rotation augmentation')
    # parser.add_argument('--normalize_scene', action='store_true',
    #                     help='rotate scene so primary pedestrian moves northwards at end of observation')
    # parser.add_argument('--augment_noise', action='store_true',
    #                     help='flag to add noise to observations for robustness')
    # parser.add_argument('--obs_dropout', action='store_true',
    #                     help='perform observation length dropout')

    ## Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ## Sequence Encoder Hyperparameters
    # hyperparameters = parser.add_argument_group('hyperparameters')
    # hyperparameters.add_argument('--hidden-dim', type=int, default=128,
    #                              help='LSTM hidden dimension')
    # hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
    #                              help='coordinate embedding dimension')
    # hyperparameters.add_argument('--pool_dim', type=int, default=256,
    #                              help='output dimension of interaction vector')
    # hyperparameters.add_argument('--goal_dim', type=int, default=64,
    #                              help='goal embedding dimension')

    # ## Grid-based pooling
    # hyperparameters.add_argument('--cell_side', type=float, default=0.6,
    #                              help='cell size of real world (in m) for grid-based pooling')
    # hyperparameters.add_argument('--n', type=int, default=12,
    #                              help='number of cells per side for grid-based pooling')
    # hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512],
    #                              help='interaction module layer dims for gridbased pooling')
    # hyperparameters.add_argument('--embedding_arch', default='one_layer',
    #                              help='interaction encoding arch for gridbased pooling')
    # hyperparameters.add_argument('--pool_constant', default=0, type=int,
    #                              help='background value (when cell empty) of gridbased pooling')
    # hyperparameters.add_argument('--norm_pool', action='store_true',
    #                              help='normalize the scene along direction of movement during grid-based pooling')
    # hyperparameters.add_argument('--front', action='store_true',
    #                              help='flag to only consider pedestrian in front during grid-based pooling')
    # hyperparameters.add_argument('--latent_dim', type=int, default=16,
    #                              help='latent dimension of encoding hidden dimension during social pooling')
    # hyperparameters.add_argument('--norm', default=0, type=int,
    #                              help='normalization scheme for input batch during grid-based pooling')

    ## Non-Grid-based pooling
    # hyperparameters.add_argument('--no_vel', action='store_true',
    #                              help='flag to not consider relative velocity of neighbours')
    # hyperparameters.add_argument('--spatial_dim', type=int, default=32,
    #                              help='embedding dimension for relative position')
    # hyperparameters.add_argument('--vel_dim', type=int, default=32,
    #                              help='embedding dimension for relative velocity')
    # hyperparameters.add_argument('--neigh', default=4, type=int,
    #                              help='number of nearest neighbours to consider')
    # hyperparameters.add_argument('--mp_iters', default=5, type=int,
    #                              help='message passing iterations in NMMP')

    ## Collision Loss
    # hyperparameters.add_argument('--col_wt', default=0., type=float,
    #                              help='collision loss weight')
    # hyperparameters.add_argument('--col_distance', default=0.2, type=float,
    #                              help='distance threshold post which collision occurs')
    args = parser.parse_args()

    # BY WJF: for ABT
    args_ABT, results_dirname = get_train_args()
    # END

    ## Set seed for reproducibility
    torch.manual_seed(args_ABT.seed)
    random.seed(args_ABT.seed)

    ## Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args_ABT.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args_ABT.path))
    
    args.output = 'OUTPUT_BLOCK/{}/Autobot_{}.pkl'.format(args_ABT.path, args_ABT.exp_id)

    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('%(message)s %(levelname)s %(name)s %(asctime)s'))
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
    # loading a previously saved model
    # args.load_state_strict = True
    # if args.nonstrict_load_state:
    #     args.load_state = args.nonstrict_load_state
    #     args.load_state_strict = False
    # if args.load_full_state:
    #     args.load_state = args.load_full_state

    # add args.device
    # args.device = torch.device('cpu')
    args.device = torch.device('cuda')

    args.path = 'DATA_BLOCK/' + args_ABT.path
    ## Prepare data
    train_scenes, train_goals, _ = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)
    val_scenes, val_goals, val_flag = prepare_data(args.path, subset='/val/', sample=args.sample, goals=args.goals)

    ## pretrained pool model (if any)
    # pretrained_pool = None

    # # create interaction/pooling modules
    # pool = None
    # if args.type == 'hiddenstatemlp':
    #     pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
    #                                  mlp_dim_vel=args.vel_dim)
    # elif args.type == 'attentionmlp':
    #     pool = AttentionMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
    #                                mlp_dim_spatial=args.spatial_dim, mlp_dim_vel=args.vel_dim)
    # elif args.type == 'nn':
    #     pool = NearestNeighborMLP(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
    # elif args.type == 'nn_lstm':
    #     pool = NearestNeighborLSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    # elif args.type == 'traj_pool':
    #     pool = TrajectronPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    # elif args.type != 'vanilla':
    #     pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
    #                             cell_side=args.cell_side, n=args.n, front=args.front,
    #                             out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
    #                             constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
    #                             norm=args.norm, layer_dims=args.layer_dims, latent_dim=args.latent_dim)

    # # create forecasting model
    # model = LSTM(pool=pool,
    #              embedding_dim=args.coordinate_embedding_dim,
    #              hidden_dim=args.hidden_dim,
    #              goal_flag=args.goals,
    #              goal_dim=args.goal_dim)

    # # optimizer and schedular
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # lr_scheduler = None
    # if args.step_size is not None:
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
    start_epoch = 0

    # # Loss Criterion
    # criterion = L2Loss(col_wt=args.col_wt, col_distance=args.col_distance) if args.loss == 'L2' \
    #                 else PredictionLoss(col_wt=args.col_wt, col_distance=args.col_distance)

    # # train
    # if args.load_state:
    #     # load pretrained model.
    #     # useful for tranfer learning
    #     print("Loading Model Dict")
    #     with open(args.load_state, 'rb') as f:
    #         checkpoint = torch.load(f)
    #     pretrained_state_dict = checkpoint['state_dict']
    #     model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

    #     if args.load_full_state:
    #     # load optimizers from last training
    #     # useful to continue model training
    #         print("Loading Optimizer Dict")
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['scheduler'])
    #         start_epoch = checkpoint['epoch']

    #trainer
    
    # trainer = Trainer(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
    #                   criterion=criterion, batch_size=args.batch_size, obs_length=args.obs_length,
    #                   pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
    #                   save_every=args.save_every, start_length=args.start_length, obs_dropout=args.obs_dropout,
    #                   augment_noise=args.augment_noise, val_flag=val_flag)
    trainer = Trainer(args_ABT, device=args.device, val_flag=val_flag)
    # trainer.loop(train_scenes, val_scenes, train_goals, val_goals, args.output, epochs=args.epochs, start_epoch=start_epoch)
    trainer.loop(train_scenes, val_scenes, train_goals, val_goals, args.output, epochs=args_ABT.num_epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
