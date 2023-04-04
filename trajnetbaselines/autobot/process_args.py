import argparse
import json
import os
from collections import namedtuple


def get_train_args():
    parser = argparse.ArgumentParser(description="AutoBots")

    # From origin TrajNet++ baselines
    parser.add_argument('--path', default='real_data', help='glob expression for data files')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    ########################################
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp',
                                 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool'),
                        help='type of interaction encoder')
    ########################################
    parser.add_argument('--save_every', default=5, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,
                        help='starting time step of encoding observation')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    ## Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    # From Autobot
    # Section: General Configuration
    parser.add_argument("--exp-id", type=str, default="train", help="Experiment identifier")

    # Section: Algorithm
    parser.add_argument("--num-modes", type=int, default=1, help="Number of discrete latent variables for Autobot.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model's hidden size.")
    parser.add_argument("--num-encoder-layers", type=int, default=1,
                        help="Number of social-temporal layers in Autobot's encoder.")
    parser.add_argument("--num-decoder-layers", type=int, default=1,
                        help="Number of social-temporal layers in Autobot's decoder.")
    parser.add_argument("--tx-hidden-size", type=int, default=384,
                        help="hidden size of transformer layers' feedforward network.")
    parser.add_argument("--tx-num-heads", type=int, default=16, help="Transformer number of heads.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout strenght used throughout model.")

    # Section: Loss Function
    parser.add_argument("--entropy-weight", type=float, default=30.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--kl-weight", type=float, default=20.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--use-FDEADE-aux-loss", type=bool, default=True,
                        help="Whether to use FDE/ADE auxiliary loss in addition to NLL (accelerates learning).")

    # Section: Training params:
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--adam-epsilon", type=float, default=1e-4, help="Adam optimiser epsilon value")
    parser.add_argument("--learning-rate-sched", type=int, nargs='+', default=[10, 20, 30, 40, 50],
                        help="Learning rate Schedule.")
    parser.add_argument("--grad-clip-norm", type=float, default=5, metavar="C", help="Gradient clipping norm")
    parser.add_argument("--num-epochs", type=int, default=102, metavar="I", help="number of iterations through the dataset.")
    args = parser.parse_args()

    return args
