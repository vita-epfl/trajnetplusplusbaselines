import os
from collections import OrderedDict
import argparse

import pickle
from joblib import Parallel, delayed
import scipy

import trajnetplusplustools
import evaluator.write as write
from evaluator.design_pd import Table

class TrajnetEvaluator:
    def __init__(self, reader_gt, scenes_gt, scenes_id_gt, scenes_sub, indexes, sub_indexes, args):
        self.reader_gt = reader_gt

        ##Ground Truth
        self.scenes_gt = scenes_gt
        self.scenes_id_gt = scenes_id_gt

        ##Prediction
        self.scenes_sub = scenes_sub

        ## Dictionary of type of trajectories
        self.indexes = indexes
        self.sub_indexes = sub_indexes

        ## The 4 types of Trajectories
        self.static_scenes = {'N': len(indexes[1])}
        self.linear_scenes = {'N': len(indexes[2])}
        self.forced_non_linear_scenes = {'N': len(indexes[3])}
        self.non_linear_scenes = {'N': len(indexes[4])}

        ## The 4 types of Interactions
        self.lf = {'N': len(sub_indexes[1])}
        self.ca = {'N': len(sub_indexes[2])}
        self.grp = {'N': len(sub_indexes[3])}
        self.others = {'N': len(sub_indexes[4])}

        ## The 4 metrics ADE, FDE, ColI, ColII
        self.average_l2 = {'N': len(scenes_gt)}
        self.final_l2 = {'N': len(scenes_gt)}

        ## Multimodal Prediction
        self.overall_nll = {'N': len(scenes_gt)}
        self.topk_ade = {'N': len(scenes_gt)}
        self.topk_fde = {'N': len(scenes_gt)}

        num_predictions = 0
        for track in self.scenes_sub[0][0]:
            if track.prediction_number and track.prediction_number > num_predictions:
                num_predictions = track.prediction_number
        self.num_predictions = num_predictions

        self.pred_length = args.pred_length
        self.obs_length = args.obs_length
        self.enable_col1 = True

        self.ade_list = {}
        self.fde_list = {}

    def aggregate(self, name, disable_collision):

        ## Overall Single Mode Scores
        average = 0.0
        final = 0.0

        ## Overall Multi Mode Scores
        average_topk_ade = 0
        average_topk_fde = 0
        average_nll = 0

        ## Aggregates ADE, FDE and Collision in GT & Pred, Topk ADE-FDE , NLL for each category & sub_category
        score = {1: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], \
                 3: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], 4: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0]}
        sub_score = {1: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], \
                     3: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], 4: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0]}

        ## Iterate
        for i in range(len(self.scenes_gt)):
            ground_truth = self.scenes_gt[i]

            ## Get Keys and Sub_keys
            keys = []
            sub_keys = []

            ## Main
            for key in list(score.keys()):
                if self.scenes_id_gt[i] in self.indexes[key]:
                    keys.append(key)
            # ## Sub
            for sub_key in list(sub_score.keys()):
                if self.scenes_id_gt[i] in self.sub_indexes[sub_key]:
                    sub_keys.append(sub_key)

            ## Extract Prediction Frames
            primary_tracks_all = [t for t in self.scenes_sub[i][0] if t.scene_id == self.scenes_id_gt[i]]
            neighbours_tracks_all = [[t for t in self.scenes_sub[i][j] if t.scene_id == self.scenes_id_gt[i]] for j in range(1, len(self.scenes_sub[i]))]

##### --------------------------------------------------- SINGLE -------------------------------------------- ####


            primary_tracks = [t for t in primary_tracks_all if t.prediction_number == 0]
            neighbours_tracks = [[t for t in neighbours_tracks_all[j] if t.prediction_number == 0] for j in range(len(neighbours_tracks_all))]

            frame_gt = [t.frame for t in ground_truth[0]][-self.pred_length:]
            frame_pred = [t.frame for t in primary_tracks]

            ## To verify if same scene
            if frame_gt != frame_pred:
                print("Frame id Groud truth: ", frame_gt)
                print("Frame id Predictions: ", frame_pred)
                raise Exception('frame numbers are not consistent')

            average_l2 = trajnetplusplustools.metrics.average_l2(ground_truth[0], primary_tracks, n_predictions=self.pred_length)
            final_l2 = trajnetplusplustools.metrics.final_l2(ground_truth[0], primary_tracks)
            self.ade_list[self.scenes_id_gt[i]] = average_l2
            self.fde_list[self.scenes_id_gt[i]] = final_l2

            if not disable_collision:
                ground_truth = self.drop_post_obs(ground_truth, self.obs_length)
                ## Collisions in GT
                # person_radius=0.1
                for j in range(1, len(ground_truth)):
                    if trajnetplusplustools.metrics.collision(primary_tracks, ground_truth[j], n_predictions=self.pred_length):
                        for key in keys:
                            score[key][2] += 1
                        ## Sub
                        for sub_key in sub_keys:
                            sub_score[sub_key][2] += 1
                        break

                ## Collision in Predictions
                # [Col-I] only if neighs in gt = neighs in prediction
                num_gt_neigh = len(ground_truth) - 1
                num_predicted_neigh = len(neighbours_tracks)
                if num_gt_neigh != num_predicted_neigh:
                    self.enable_col1 = False
                    for key in score:
                        score[key][4] = 0
                        score[key][3] = 0
                    for sub_key in sub_score:
                        sub_score[sub_key][4] = 0
                        sub_score[sub_key][3] = 0

                if self.enable_col1:
                    for key in keys:
                        score[key][4] += 1
                        for j in range(len(neighbours_tracks)):
                            if trajnetplusplustools.metrics.collision(primary_tracks, neighbours_tracks[j], n_predictions=self.pred_length):
                                score[key][3] += 1
                                break
                    ## Sub
                    for sub_key in sub_keys:
                        sub_score[sub_key][4] += 1
                        for j in range(len(neighbours_tracks)):
                            if trajnetplusplustools.metrics.collision(primary_tracks, neighbours_tracks[j], n_predictions=self.pred_length):
                                sub_score[sub_key][3] += 1
                                break


            # aggregate FDE and ADE
            average += average_l2
            final += final_l2

            for key in keys:
                score[key][0] += average_l2
                score[key][1] += final_l2

            ## Sub
            for sub_key in sub_keys:
                sub_score[sub_key][0] += average_l2
                sub_score[sub_key][1] += final_l2

##### --------------------------------------------------- SINGLE -------------------------------------------- ####

##### --------------------------------------------------- Top 3 -------------------------------------------- ####

            if self.num_predictions > 1:
                topk_ade, topk_fde = trajnetplusplustools.metrics.topk(primary_tracks_all, ground_truth[0], n_predictions=self.pred_length)

                average_topk_ade += topk_ade
                ##Key
                for key in keys:
                    score[key][5] += topk_ade
                ## SubKey
                for sub_key in sub_keys:
                    sub_score[sub_key][5] += topk_ade

                average_topk_fde += topk_fde
                ##Key
                for key in keys:
                    score[key][6] += topk_fde
                ## SubKey
                for sub_key in sub_keys:
                    sub_score[sub_key][6] += topk_fde

##### --------------------------------------------------- Top 3 -------------------------------------------- ####

##### --------------------------------------------------- NLL -------------------------------------------- ####
            if self.num_predictions > 48:
                nll = trajnetplusplustools.metrics.nll(primary_tracks_all, ground_truth[0], n_predictions=self.pred_length, n_samples=50)

                average_nll += nll
                ##Key
                for key in keys:
                    score[key][7] += nll
                ## SubKey
                for sub_key in sub_keys:
                    sub_score[sub_key][7] += nll
##### --------------------------------------------------- NLL -------------------------------------------- ####

        ## Average ADE and FDE
        average /= len(self.scenes_gt)
        final /= len(self.scenes_gt)

        ## Average TopK ADE and Topk FDE and NLL
        average_topk_ade /= len(self.scenes_gt)
        average_topk_fde /= len(self.scenes_gt)
        average_nll /= len(self.scenes_gt)

        ## Average categories
        for key in list(score.keys()):
            if self.indexes[key]:
                score[key][0] /= len(self.indexes[key])
                score[key][1] /= len(self.indexes[key])

                score[key][5] /= len(self.indexes[key])
                score[key][6] /= len(self.indexes[key])
                score[key][7] /= len(self.indexes[key])

        ## Average subcategories
        ## Sub
        for sub_key in list(sub_score.keys()):
            if self.sub_indexes[sub_key]:
                sub_score[sub_key][0] /= len(self.sub_indexes[sub_key])
                sub_score[sub_key][1] /= len(self.sub_indexes[sub_key])

                sub_score[sub_key][5] /= len(self.sub_indexes[sub_key])
                sub_score[sub_key][6] /= len(self.sub_indexes[sub_key])
                sub_score[sub_key][7] /= len(self.sub_indexes[sub_key])

        # ##Adding value to dict
        self.average_l2[name] = average
        self.final_l2[name] = final

        ##APPEND to overall keys
        self.overall_nll[name] = average_nll
        self.topk_ade[name] = average_topk_ade
        self.topk_fde[name] = average_topk_fde

        ## Main
        self.static_scenes[name] = score[1]
        self.linear_scenes[name] = score[2]
        self.forced_non_linear_scenes[name] = score[3]
        self.non_linear_scenes[name] = score[4]

        ## Sub_keys
        self.lf[name] = sub_score[1]
        self.ca[name] = sub_score[2]
        self.grp[name] = sub_score[3]
        self.others[name] = sub_score[4]

        return self

    def result(self):
        return self.average_l2, self.final_l2, \
               self.static_scenes, self.linear_scenes, self.forced_non_linear_scenes, self.non_linear_scenes, \
               self.lf, self.ca, self.grp, self.others, \
               self.topk_ade, self.topk_fde, self.overall_nll

    def save_distance_lists(self, input_file):
        distance_file = os.path.dirname(input_file).replace('test_pred', 'ade_fde_list')
        os.makedirs(distance_file)
        with open(distance_file + '/ade_fde.pkl', 'wb') as handle:
            pickle.dump([self.ade_list, self.fde_list], handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## drop pedestrians that appear post observation
    def drop_post_obs(self, ground_truth, obs_length):
        obs_end_frame = ground_truth[0][obs_length].frame
        ground_truth = [track for track in ground_truth if track[0].frame < obs_end_frame]
        return ground_truth

def collision_test(list_sub, name, args):
    """ Simple Collision Test """
    submit_datasets = [args.path + name + '/' + f for f in list_sub if 'collision_test.ndjson' in f]
    if len(submit_datasets):
        # Scene Prediction
        reader_sub = trajnetplusplustools.Reader(submit_datasets[0], scene_type='paths')
        scenes_sub = [s for _, s in reader_sub.scenes()]

        if trajnetplusplustools.metrics.collision(scenes_sub[0][0], scenes_sub[0][1], n_predictions=args.pred_length):
            return "Fail"
        return "Pass"

    return "NA"

def eval(gt, input_file, args):
    # Ground Truth
    reader_gt = trajnetplusplustools.Reader(gt, scene_type='paths')
    scenes_gt = [s for _, s in reader_gt.scenes()]
    scenes_id_gt = [s_id for s_id, _ in reader_gt.scenes()]

    # Scene Predictions
    reader_sub = trajnetplusplustools.Reader(input_file, scene_type='paths')
    scenes_sub = [s for _, s in reader_sub.scenes()]

    ## indexes is dictionary deciding which scenes are in which type
    indexes = {}
    for i in range(1, 5):
        indexes[i] = []
    ## sub-indexes
    sub_indexes = {}
    for i in range(1, 5):
        sub_indexes[i] = []
    for scene in reader_gt.scenes_by_id:
        tags = reader_gt.scenes_by_id[scene].tag
        main_tag = tags[0:1]
        sub_tags = tags[1]
        for ii in range(1, 5):
            if ii in main_tag:
                indexes[ii].append(scene)
            if ii in sub_tags:
                sub_indexes[ii].append(scene)

    # Evaluate
    evaluator = TrajnetEvaluator(reader_gt, scenes_gt, scenes_id_gt, scenes_sub, indexes, sub_indexes, args)
    evaluator.aggregate('kf', args.disable_collision)

    ## Save Lists
    # evaluator.save_distance_lists(input_file)

    return evaluator.result()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='trajdata',
                        help='directory of data to test')
    parser.add_argument('--output', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--write_only', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--sf', action='store_true',
                        help='consider socialforce in evaluation')
    parser.add_argument('--orca', action='store_true',
                        help='consider orca in evaluation')
    parser.add_argument('--kf', action='store_true',
                        help='consider kalman in evaluation')
    parser.add_argument('--cv', action='store_true',
                        help='consider constant velocity in evaluation')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    args = parser.parse_args()

    scipy.seterr('ignore')

    args.output = args.output if args.output is not None else []
    ## assert length of output models is not None
    if (not args.sf) and (not args.orca) and (not args.kf) and (not args.cv):
        assert len(args.output), 'No output file is provided'

    ## Path to the data folder name to predict
    args.path = 'DATA_BLOCK/' + args.path + '/'

    ## Test_pred : Folders for saving model predictions
    args.path = args.path + 'test_pred/'

    ## Writes to Test_pred
    ## Does NOT overwrite existing predictions if they already exist ###
    write.main(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluates test_pred with test_private
    names = []
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)
        names.append(model_name)

    ## labels
    if args.labels:
        labels = args.labels
    else:
        labels = names

    # Initiate Result Table
    table = Table()

    for num, name in enumerate(names):
        print(name)

        result_file = args.path.replace('pred', 'results') + name

        ## If result was pre-calculated and saved, Load
        if os.path.exists(result_file + '/results.pkl'):
            print("Loading Saved Results")
            with open(result_file + '/results.pkl', 'rb') as handle:
                [final_result, sub_final_result, col_result] = pickle.load(handle)
            table.add_result(labels[num], final_result, sub_final_result)
            table.add_collision_entry(labels[num], col_result)

        # ## Else, Calculate results and save
        else:
            list_sub = sorted([f for f in os.listdir(args.path + name)
                               if not f.startswith('.')])

            ## Simple Collision Test
            col_result = collision_test(list_sub, name, args)
            table.add_collision_entry(labels[num], col_result)

            submit_datasets = [args.path + name + '/' + f for f in list_sub if 'collision_test.ndjson' not in f]
            true_datasets = [args.path.replace('pred', 'private') + f for f in list_sub if 'collision_test.ndjson' not in f]

            ## Evaluate submitted datasets with True Datasets [The main eval function]
            # results = {submit_datasets[i].replace(args.path, '').replace('.ndjson', ''):
            #             eval(true_datasets[i], submit_datasets[i], args)
            #            for i in range(len(true_datasets))}

            results_list = Parallel(n_jobs=4)(delayed(eval)(true_datasets[i], submit_datasets[i], args)
                                                            for i in range(len(true_datasets)))
            results = {submit_datasets[i].replace(args.path, '').replace('.ndjson', ''): results_list[i] 
                       for i in range(len(true_datasets))}

            # print(results)
            ## Generate results
            final_result, sub_final_result = table.add_entry(labels[num], results)

            ## Save results as pkl (to avoid computation again)
            os.makedirs(result_file)
            with open(result_file + '/results.pkl', 'wb') as handle:
                pickle.dump([final_result, sub_final_result, col_result], handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## Make Result Table
    table.print_table()

if __name__ == '__main__':
    main()
