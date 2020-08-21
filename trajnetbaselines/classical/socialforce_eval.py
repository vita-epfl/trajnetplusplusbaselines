"""Command line tool to create a table of evaluations metrics."""
import argparse

import pickle

import trajnetplusplustools
from trajnetplusplustools import show
from trajnetplusplustools.interactions import collision_avoidance

from . import kalman
from . import socialforce
from . import orca

class Evaluator(object):
    def __init__(self, scenes, dest_dict=None, params=None, args=None):
        self.scenes = scenes
        self.dest = dest_dict
        self.params = params

        self.average_l2 = {'N': len(scenes)}
        self.final_l2 = {'N': len(scenes)}

        self.args = args

    def aggregate(self, name, predictor, dest_type='true'):
        print('evaluating', name)

        average = 0.0
        final = 0.0

        # pred_dict = {}
        # pred_neigh_dict = {}
        # n = 0

        for _, paths in enumerate(self.scenes):
            ## select only those trajectories which interactions ##
            # rows = trajnetplusplustools.Reader.paths_to_xy(paths)
            # neigh_paths = paths[1:]
            # interaction_index = collision_avoidance(rows)
            # neigh = list(compress(neigh_paths, interaction_index))
            # paths = [paths[0]] + neigh

            if 'kf' in name:
                prediction, neigh = predictor(paths, n_predict=self.args.pred_length, obs_length=self.args.obs_length)[0]
            if 'sf' in name:
                prediction, neigh = predictor(paths, self.dest, dest_type, self.params['sf'], args=self.args)[0]
            if 'orca' in name:
                prediction, neigh = predictor(paths, self.dest, dest_type, self.params['orca'], args=self.args)[0]

            ## visualize predictions ##
            # pred_dict['pred'] = prediction
            # pred_neigh_dict['pred'] = neigh
            # n += 1
            # if n < 17:
            #     with show.predicted_paths(paths, pred_dict, pred_neigh_paths=pred_neigh_dict):
            #         pass
            # else:
            #     break

            ## Convert numpy array to Track Rows ##
            ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
            observed_path = paths[0]
            frame_diff = observed_path[1].frame - observed_path[0].frame
            first_frame = observed_path[self.args.obs_length-1].frame + frame_diff
            ped_id = observed_path[0].pedestrian

            ## make Track Rows
            prediction = [trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id, prediction[i, 0], prediction[i, 1], 0)
                          for i in range(len(prediction))]

            average_l2 = trajnetplusplustools.metrics.average_l2(paths[0], prediction)
            final_l2 = trajnetplusplustools.metrics.final_l2(paths[0], prediction)

            # aggregate
            average += average_l2
            final += final_l2

        average /= len(self.scenes)
        final /= len(self.scenes)

        self.average_l2[name] = average
        self.final_l2[name] = final

        return self

    def result(self):
        return self.average_l2, self.final_l2


def eval(input_file, dest_file, simulator, params, type_ids, args):
    print('dataset', input_file)

    reader = trajnetplusplustools.Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]

    ## Filter scenes according to category type
    # if type_ids is None:
    #     trajectory_type = 3
    #     interaction_type = 2
    #     type_ids = [scene_id for scene_id in reader.scenes_by_id \
    #                 if interaction_type in reader.scenes_by_id[scene_id].tag[1]]

    # scenes = [scenes[type_id] for type_id in type_ids]

    ## If final destination of pedestrian provided
    dest_dict = None
    if dest_file is not None:
        dest_dict = pickle.load(open(dest_file, "rb"))

    evaluator = Evaluator(scenes, dest_dict, params, args)

    ## Evaluate all
    if simulator == 'all':
        for dest_type in ['interp']:
            evaluator.aggregate('orca' + dest_type, orca.predict, dest_type)
            evaluator.aggregate('sf' + dest_type, socialforce.predict, dest_type)
        evaluator.aggregate('kf', kalman.predict)

    # ORCA only
    elif simulator == 'orca':
        for dest_type in ['interp']:
            evaluator.aggregate('orca' + dest_type, orca.predict, dest_type)

    # Social Force only
    elif simulator == 'sf':
        for dest_type in ['interp']:
            evaluator.aggregate('sf' + dest_type, socialforce.predict, dest_type)

    # Kalman only
    elif simulator == 'kf':  
        evaluator.aggregate('kf', kalman.predict)

    return evaluator.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')

    parser.add_argument('--simulator', default='all',
                        choices=('all', 'orca', 'sf', 'kalman'))

    parser.add_argument('--tau', default=0.5, type=float,
                        help='Tau of Social Force')
    parser.add_argument('--vo', default=2.1, type=float,
                        help='V0 of Social Force')
    parser.add_argument('--sigma', default=0.3, type=float,
                        help='sigma of Social Force')

    parser.add_argument('--min_dist', default=4, type=float,
                        help='MinNeighDist of ORCA')
    parser.add_argument('--react_time', default=4, type=float,
                        help='NeighReactTime of ORCA')
    parser.add_argument('--radius', default=0.6, type=float,
                        help='agent radius of ORCA')

    args = parser.parse_args()

    params = {}
    if args.simulator == 'sf' or 'all':
        params['sf'] = [args.tau, args.vo, args.sigma]
    if args.simulator == 'orca' or 'all':
        params['orca'] = [args.min_dist, args.react_time, args.radius]

    print(params)
    datasets = [
        'DATA_BLOCK/data/train/real_data/biwi_hotel.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_zara01.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_zara03.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_students001.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_students003.ndjson',
        # 'DATA_BLOCK/data/train/real_data/lcas.ndjson',
        # 'DATA_BLOCK/data/train/real_data/wildtrack.ndjson',
    ]

    ## Final Destination Dictionaries
    dest_dicts = [
        'goal_files/biwi_hotel.pkl',
        # 'goal_files/crowds_zara01.pkl',
        # 'goal_files/crowds_zara03.pkl',
        # 'goal_files/crowds_students001.pkl',
        # 'goal_files/crowds_students003.pkl',
        # 'goal_files/lcas.pkl',
        # 'goal_files/wildtrack.pkl',   
    ]

    ## Selected Interaction Scenes for Fitting and Visualizing Simulator Parameters
    filtered_ids = {}
    ## IDs according to TrajData
    # filtered_ids['biwi_hotel'] = [92, 116, 118, 121, 123, 174, 233] # 102, 103, 175, 176
    # filtered_ids['crowds_zara01'] = [13, 107, 108, 115, 117, 119, 120, 225, 226, 270, 320, 321, \
    #                                  339, 345, 367, 390, 394, 395, 396, 397, 411, 586, 587, 748, \
    #                                  750, 751, 789, 806, 807, 824, 837, 838, 839, 840, 929, 931]
    # filtered_ids['crowds_zara03'] = [46, 47, 79, 80, 97, 200, 202, 256, 257, 258, 342, 343, 368, \
    #                                  449, 447, 450, 590, 703, 704, 755, 772, 855, 856, 877, 893]

    # filtered_ids['biwi_hotel'] = [233] # 103, 116
    # filtered_ids['crowds_zara01'] = [824, 930]
    # filtered_ids['crowds_zara03'] = [772]

    results = {}
    for i, dataset in enumerate(datasets):
        type_ids = None
        dataset_name = dataset.replace('DATA_BLOCK/data/train/real_data/', '').replace('.ndjson', '')
        if dataset_name in filtered_ids:
            type_ids = filtered_ids[dataset_name]       
        results[dataset_name] = eval(dataset, dest_dicts[i], args.simulator, params, type_ids, args)

    if args.simulator == 'all':
        print('## Average L2 [m]')
        print('{dataset:>30s} |   N  | ORCA | SF | KF '.format(dataset=''))
        for dataset, (r, _) in results.items():
            print(
                '{dataset:>30s}'
                ' | {r[N]:>4}'
                ' | {r[orcainterp]:.2f}'
                ' | {r[sfinterp]:.2f}'
                ' | {r[kf]:.2f}'.format(dataset=dataset, r=r)
            )
        print('')

        print('## Final L2 [m]')
        print('{dataset:>30s} |   N  | ORCA | SF | KF '.format(dataset=''))
        for dataset, (_, r) in results.items():
            print(
                '{dataset:>30s}'
                ' | {r[N]:>4}'
                ' | {r[orcainterp]:.2f}'
                ' | {r[sfinterp]:.2f}'
                ' | {r[kf]:.2f}'.format(dataset=dataset, r=r)
            )

    ## For Hyperparameter Tuning
    # print('params: {}, {}, {} \n'.format(*params['sf']))
    # with open(args.simulator + "_final.txt", "a") as myfile:
    #         myfile.write('params: {}, {}, {} \n'.format(*params['sf']))
    #         myfile.write('## Average L2 [m]\n')
    #         myfile.write('{dataset:>30s} |   N  | Int \n'.format(dataset=''))
    #         for dataset, (r, _) in results.items():
    #             myfile.write(
    #                         '{dataset:>30s}'
    #                         ' | {r[N]:>4}'
    #                         ' | {r[sfinterp]:.2f} \n'.format(dataset=dataset, r=r)
    #             )

    #         myfile.write('\n')
    #         myfile.write('## Final L2 [m] \n')
    #         myfile.write('{dataset:>30s} |   N  | Int \n'.format(dataset=''))
    #         for dataset, (_, r) in results.items():
    #             myfile.write(
    #                         '{dataset:>30s}'
    #                         ' | {r[N]:>4}'
    #                         ' | {r[sfinterp]:.2f} \n'.format(dataset=dataset, r=r)
    #             )
    #         myfile.write('\n \n \n')

if __name__ == '__main__':
    main()
