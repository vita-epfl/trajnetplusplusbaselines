import numpy as np
import trajnettools
import shutil
import os
import warnings
import evaluator.write as write
import argparse

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from collections import OrderedDict
from trajnetbaselines import kalman

class TrajnetEvaluator:
    def __init__(self, reader_gt, scenes_gt, scenes_id_gt, scenes_sub, indexes):
        self.reader_gt = reader_gt
        
        ##Ground Truth
        self.scenes_gt = scenes_gt
        self.scenes_id_gt = scenes_id_gt

        ##Prediction
        self.scenes_sub = scenes_sub

        ## Dictionary of type of trajectories
        self.indexes = indexes

        ## The 4 types of Trajectories
        self.static_scenes = {'N': len(indexes[1])}
        self.linear_scenes = {'N': len(indexes[2])}
        self.forced_non_linear_scenes = {'N': len(indexes[3])}
        self.non_linear_scenes = {'N': len(indexes[4])}

        ## The 4 metrics ADE, FDE, ColI, ColII
        self.average_l2 = {'N': len(scenes_gt)}
        self.final_l2 = {'N': len(scenes_gt)}
        self.final_collision = {'N': len(scenes_gt)}
        self.prediction_collision = {'N': len(scenes_gt)}

    def aggregate(self, name, disable_collision):

        ## Overall Scores
        average = 0.0
        final = 0.0
        glob_collision = 0
        prediction_collision = 0

        ## Aggregates ADE, FDE and Collision in GT & Pred for each category 
        score = {1: [0.0, 0.0, 0, 0], 2: [0.0, 0.0, 0, 0], 3: [0.0, 0.0, 0, 0], 4: [0.0, 0.0, 0, 0]}

        if self.scenes_sub: 
            ## Number of future trajectories proposed by the model #Multimodality
            num_predictions = 0
            tmp_prediction = {}
            for track in self.scenes_sub[0][0]:
                if track.prediction_number and track.prediction_number > num_predictions:
                    num_predictions = scene.prediction_number
            ## Max. 3 trajectories can only be outputted
            if num_predictions > 2:
                warnings.warn("3 predictions at most")
                num_predictions = 2

        ## Iterate
        for i in range(len(self.scenes_gt)):
            ground_truth = self.scenes_gt[i]
            
            if self.scenes_sub: 
                ## Extract Prediction Frames
                primary_tracks = [t for t in self.scenes_sub[i][0] if t.scene_id == self.scenes_id_gt[i]]
                neighbours_tracks = [[t for t in self.scenes_sub[i][j] if t.scene_id == self.scenes_id_gt[i]] for j in range(1, len(self.scenes_sub[i]))]
                # print(primary_tracks)
                # print(neighbours_tracks)

                l2 = 1e10
                for np in range(num_predictions + 1):
                    primary_prediction = [t for t in primary_tracks if t.prediction_number == np]
                    tmp_score = trajnettools.metrics.final_l2(ground_truth[0], primary_prediction)
                    if tmp_score < l2:
                        best_prediction_number = np
                        l2 = tmp_score

                primary_tracks = [t for t in primary_tracks if t.prediction_number == best_prediction_number]
                neighbours_tracks = [[t for t in neighbours_tracks[j] if t.prediction_number == best_prediction_number] for j in range(len(neighbours_tracks))]

                frame_gt = [t.frame for t in ground_truth[0]][-12:]
                frame_pred = [t.frame for t in primary_tracks]
                # print(ground_truth[0])
                # print(primary_tracks)
                ## To verify if talking about same scene
                if frame_gt != frame_pred:
                    raise Exception('frame numbers are not consistent')
            else:
                ## Extract Prediction Frames
                primary_tracks, neighbours_tracks = kalman.predict(ground_truth)

            average_l2 = trajnettools.metrics.average_l2(ground_truth[0], primary_tracks)
            final_l2 = trajnettools.metrics.final_l2(ground_truth[0], primary_tracks)
            
            if not disable_collision:
                ## Collisions in GT
                for j in range(1, len(ground_truth)):
                    if trajnettools.metrics.collision(primary_tracks, ground_truth[j]):
                        glob_collision += 1
                        for key in list(score.keys()):
                            if self.scenes_id_gt[i] in self.indexes[key]:
                                score[key][2] += 1
                        break

                ## Collision in Predictions 
                for j in range(len(neighbours_tracks)):
                    if trajnettools.metrics.collision(primary_tracks, neighbours_tracks[j]):
                        prediction_collision += 1
                        for key in list(score.keys()):
                            if self.scenes_id_gt[i] in self.indexes[key]:
                                score[key][3] += 1
                        break

            # aggregate FDE and ADE
            average += average_l2
            final += final_l2
            for key in list(score.keys()):
                if self.scenes_id_gt[i] in self.indexes[key]:
                    score[key][0] += average_l2
                    score[key][1] += final_l2

        #print(index_collision, len(index_collision))
        ## Average ADE and FDE
        average /= len(self.scenes_gt)
        final /= len(self.scenes_gt)
        for key in list(score.keys()):
            if self.indexes[key]:
                score[key][0] /= len(self.indexes[key])
                score[key][1] /= len(self.indexes[key])

        ##Adding value to dict
        self.average_l2[name] = average
        self.final_l2[name] = final
        self.final_collision[name] = glob_collision
        self.prediction_collision[name] = prediction_collision

        self.static_scenes[name] = score[1]
        self.linear_scenes[name] = score[2]
        self.forced_non_linear_scenes[name] = score[3]
        self.non_linear_scenes[name] = score[4]


        return self

    def result(self):
        return self.average_l2, self.final_l2, self.static_scenes, self.linear_scenes, \
               self.forced_non_linear_scenes, self.non_linear_scenes, self.final_collision,\
               self.prediction_collision


def eval(gt, input_file, disable_collision, args):
    
    # Ground Truth
    reader_gt = trajnettools.Reader(gt, scene_type='paths')
    scenes_gt = [s for _, s in reader_gt.scenes()]
    scenes_id_gt = [s_id for s_id, _ in reader_gt.scenes()]

    ## indexes is dictionary deciding which scenes are in which type
    indexes = {}
    for i in range(1,5):
        indexes[i] = []
    for scene in reader_gt.scenes_by_id:
        # print("Scene:", scene)
        for ii in range(1, 5):
            # if reader_gt.scenes_by_id[scene].tag == ii:
            if ii in reader_gt.scenes_by_id[scene].tag:
                indexes[ii].append(scene)

    if input_file:
        # Scene Predictions
        reader_sub = trajnettools.Reader(input_file, scene_type='paths')
        scenes_sub = [s for _, s in reader_sub.scenes()]
    else:
        scenes_sub = None

    evaluator = TrajnetEvaluator(reader_gt, scenes_gt, scenes_id_gt, scenes_sub, indexes)
    evaluator.aggregate('kf', disable_collision)

    return evaluator.result()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='syn_data', choices = ('syn_data','clean_controlled', 'noisy_controlled', 'mix_synth'),
                        help='path of data')
    parser.add_argument('--output', required=True, nargs='+',
                        help='output folder')
    parser.add_argument('--disable-write', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metric')
    parser.add_argument('--test', default = True)
    parser.add_argument('--test-path', default = 'test')
    args = parser.parse_args()

    ## Path to the data folder name to predict 
    args.data = 'DATA_BLOCK/' + args.data + '/'

    ## Train_pred or Test_pred : Folders for saving model predictions
    args.data = args.data + args.test_path + '_pred/'

    ## Writes to Test_pred
    ########## Does this overwrite existing predictions? No. ####################
    if not args.disable_write:
        write.main(args)


    ## Evaluates test_pred with test_private
    names = []
    for model in args.output:
        j = -1
        while model[j] != '/':
            j -= 1
        names.append(model[j+1:].replace('.pkl', ''))

    print("NAMES: ", names)
    names.append('kalman')
    
    overall = {}
    for name in names:
        if name != 'kalman':
            list_sub = sorted([f for f in os.listdir(args.data + name)
                               if not f.startswith('.')])

            submit_datasets = [args.data + name + '/' + f for f in list_sub]
            true_datasets = [args.data.replace('pred', 'private') + f for f in list_sub]

            print(name)
            print("Submit Datasets:", submit_datasets)
            print("True Datasets:", true_datasets)

            # print(submit_datasets[i].replace(args.data, '').replace('.ndjson', ''))
            ## Evaluate submitted datasets with True Datasets [The main eval function]
            results = {submit_datasets[i].replace(args.data, '').replace('.ndjson', ''):
                       eval(true_datasets[i], submit_datasets[i], args.disable_collision, args)
                       for i in range(len(true_datasets))}
        else:
            results = {'kalman': 
                       eval(true_datasets[i], None, args.disable_collision, args)
                       for i in range(len(true_datasets))}            

        print("Results: ", results)
        ## Give result stats to overall dict
        overall[name] = {}
        overall[name]['o_all'] = np.array([0, 0.0, 0.0, 0.0, 0.0])
        overall[name]['1'] = np.array([0, 0.0, 0.0, 0.0, 0.0])
        overall[name]['2'] = np.array([0, 0.0, 0.0, 0.0, 0.0])
        overall[name]['3'] = np.array([0, 0.0, 0.0, 0.0, 0.0])
        overall[name]['4'] = np.array([0, 0.0, 0.0, 0.0, 0.0])
        for dataset, (a, b, c, d, e, f, g, h) in results.items():
            overall[name]['o_all'] += np.array([a['N'], a['kf'] * a['N'], b['kf'] * a['N'], g['kf'],
                                                h['kf']])
            overall[name]['1'] += np.array([c['N'], c['kf'][0] * c['N'], c['kf'][1] * c['N'],
                                            c['kf'][2], c['kf'][3]])
            overall[name]['2'] += np.array([d['N'], d['kf'][0] * d['N'], d['kf'][1] * d['N'],
                                            d['kf'][2], d['kf'][3]])
            overall[name]['3'] += np.array([e['N'], e['kf'][0] * e['N'], e['kf'][1] * e['N'],
                                            e['kf'][2], e['kf'][3]])
            overall[name]['4'] += np.array([f['N'], f['kf'][0] * f['N'], f['kf'][1] * f['N'],
                                            f['kf'][2], f['kf'][3]])
        print('')

    print('')
    ## Display Results
    print('Results by method')
    print('{name:>35s}'.format(name=''))
    print(
        '{name:>35s} |      |                 Overall.          |'
        '              II                   |                 III               |                 IV                |'
            .format(name=''))
    print(
        '{name:>35s} | grade|   N  |  AVG |  FNL |  Col  | Col2 |'
        '   N  |  AVG |  FNL |  Col  | Col2 |   N  |  AVG |  FNL |  Col  | Col2 |   N  |  AVG |  FNL |  Col  | Col2 |'
            .format(name=''))

    for name in list(overall.keys()):
        final_results = []
        for keys in list(overall[name].keys()):
            if overall[name][keys][0] != 0:
                overall[name][keys][1] /= overall[name][keys][0]
                overall[name][keys][2] /= overall[name][keys][0]
                overall[name][keys][3] /= (overall[name][keys][0]*0.01)
                overall[name][keys][4] /= (overall[name][keys][0]* 0.01)
                final_results += [int(overall[name][keys][0]), overall[name][keys][1], overall[name][keys][2], overall[name][keys][3], overall[name][keys][4]]
            else:
                final_results += [0, 0.0, 0.0, 0.0, 0.0]

        global_grade = final_results[2]
        print(
            '{dataset:>35s}'
            ' | {global_grade:.2f}'
            ' | {final_results[0]:>4}'
            ' | {final_results[1]:.2f}'
            ' | {final_results[2]:.2f}'
            ' | {final_results[3]:.1f}'
            '  | {final_results[4]: .1f}'
            ' | {final_results[10]:>4}'
            ' | {final_results[11]:.2f}'
            ' | {final_results[12]:.2f}'
            ' | {final_results[13]:.1f}'
            '  | {final_results[14]:.1f}'
            ' | {final_results[15]:>4}'
            ' | {final_results[16]:.2f}'
            ' | {final_results[17]:.2f}'
            ' | {final_results[18]:.1f}'
            '  | {final_results[19]:.1f}'
            ' | {final_results[20]:>4}'
            ' | {final_results[21]:.2f}'
            ' | {final_results[22]:.2f}'
            ' | {final_results[23]:.1f}'
            '  | {final_results[24]:.1f} |'
            .format(dataset=name, global_grade=global_grade, final_results=final_results))

        # final_results = []
        # final_results += [int(overall[name]['o_all'][0]),                               ## N
        #                   overall[name]['o_all'][1] / overall[name]['o_all'][0],        ## ADE
        #                   overall[name]['o_all'][2] / overall[name]['o_all'][0],        ## FDE
        #                   int(overall[name]['o_all'][3]) / overall[name]['o_all'][0],   ## GT Col
        #                   int(overall[name]['o_all'][4]) / overall[name]['o_all'][0]]   ## Pred Col

        # for i in range(1, 5):
        #     if overall[name][str(i)][0] == 0:
        #         final_results += [0.0, 0.0, 0.0]
        #     else:
        #         final_results += [int(overall[name][str(i)][0]), overall[name][str(i)][1] / overall[name][str(i)][0],
        #                           overall[name][str(i)][2] / overall[name][str(i)][0]]

        # radar_chart(name, final_results, 'real', prediction_col)
# def radar_chart(name, final_results, type_, prediction_col):
#     # Set data

#     if 'real' in type_:
#         type_ = 'real'
        
#         # s_lstm = [1.03, 1.84, 27.90, 4.38, 0.47, 0.92, 0.60, 1.21, 1.11, 1.95, 1.05, 1.90]
#         # min_values = [1.95, 1.84, 2.40, 26.71, 1.03]
#         # s_lstm_values = [1.95, 1.84, 4.38, 27.90, 1.03]

#         s_lstm = [0.19, 0.24, 0.53, 23.98, 0.00, 0.00, 0.11, 0.17, 0.18, 0.22, 0.36, 0.64]
#         min_values = [0.22, 0.24, 23.98, 0.53, 0.19]
#         s_lstm_values = [0.22, 0.24, 23.98, 0.53, 0.19]   

#     else:
#         type_ = 'synth'

#         # s_lstm = [0.27, 0.49, 8.20, 17.61, 0.00, 0.00, 0.08, 0.14, 0.29, 0.53, 0.20, 0.36]
#         # min_values = [0.53, 0.49, 10.05, 7.42, 0.27]
#         # s_lstm_values = [0.53, 0.49, 17.61, 8.20, 0.27]

#         s_lstm = [0.27, 0.49, 8.20, 17.61, 0.00, 0.00, 0.08, 0.14, 0.29, 0.53, 0.20, 0.36]
#         min_values = [0.53, 0.49, 10.05, 7.42, 0.27]
#         s_lstm_values = [0.53, 0.49, 17.61, 8.20, 0.27]

#     sub_model = [final_results[1], final_results[2], 100 * final_results[3] / final_results[0],
#                  100 * final_results[4] / prediction_col, final_results[6], final_results[7],
#                  final_results[9], final_results[10], final_results[12], final_results[13],
#                  final_results[15], final_results[16]]

#     fde_3 = min(min_values[0], sub_model[9])
#     fde = min(min_values[1], sub_model[1])
#     pred_col = min(min_values[2], sub_model[3])
#     gt_col = min(min_values[3], max(sub_model[2], 0.01))
#     ade = min(min_values[4], sub_model[0])
#     df = pd.DataFrame(OrderedDict({
#         'group': ['O-LSTM', name],
#         'FDE for type III': [fde_3 / s_lstm_values[0], fde_3 / sub_model[9]],
#         'FDE': [fde / s_lstm_values[1], fde / sub_model[1]],
#         'Pred. Colision': [pred_col / s_lstm_values[2], pred_col / sub_model[3]],
#         'GT Colision': [gt_col / s_lstm_values[3], gt_col / max(sub_model[2], 0.01)],
#         'ADE': [ade / s_lstm_values[4], ade / sub_model[0]]
#     }))

#     # ------- PART 1: Create background

#     # number of variable
#     categories = list(df)[1:]
#     N = len(categories)

#     # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
#     angles = [n / float(N) * 2 * np.pi for n in range(N)]
#     angles += angles[:1]

#     # Initialise the spider plot
#     f = plt.figure(figsize=(12, 12))
#     ax = f.add_subplot(111, polar=True)

#     # If you want the first axis to be on top:
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)

#     # Draw one axe per variable + add labels labels yet
#     plt.xticks(angles[:-1], categories)

#     # Draw ylabels
#     ax.set_rlabel_position(0)
#     plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], color="grey", size=7)
#     plt.ylim(0, 1.1)

#     # ------- PART 2: Add plots

#     # Plot each individual = each line of the data
#     # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

#     # Ind1
#     values = df.loc[0].drop('group').values.flatten().tolist()
#     values += values[:1]
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label="occupancy_35_cell_6")
#     ax.fill(angles, values, 'b', alpha=0.1)

#     # Ind2
#     values = df.loc[1].drop('group').values.flatten().tolist()
#     values += values[:1]
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label=name)
#     ax.fill(angles, values, 'r', alpha=0.1)

#     cellText = [['ADE', 'FDE', 'Col 1', 'Col 2', 'ADE', 'FDE', 'ADE', 'FDE', 'ADE', 'FDE',
#                  'ADE', 'FDE'],
#                 s_lstm,
#                 [sub_model[index].__format__('.2f') for index in range(12)]]
#     rowLabels = ['', 'S-LSTM', name]
#     colLabels = ['Overall', 'O.', 'O.', 'O.', 'I', 'I', 'II', 'II', 'III', 'III',
#                  'IV', 'IV']

#     the_table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels,
#                           cellLoc='center')

#     the_table.auto_set_font_size(False)
#     the_table.set_fontsize(12)
#     the_table.scale(1, 2)

#     for (row, col), cell in the_table.get_celld().items():
#         # cell._text.set_color('white')
#         if (row == 0) or (row == 1) or (col == -1):
#             cell.set_text_props(fontproperties=FontProperties(weight='bold'))

#     # Add legend
#     plt.legend(loc='upper right')  # , bbox_to_anchor=(0.1, 0.1)
#     plt.savefig('radar_chart_' + name + type_ +'.png', bbox_inches='tight')
#     plt.show()
#     plt.close()

if __name__ == '__main__':
    main()

