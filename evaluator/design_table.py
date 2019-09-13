import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

class Table(object):
    """docstring for Table"""
    def __init__(self, arg=None):
        super(Table, self).__init__()

        self.entries = {}
        self.sub_entries = {}
        self.arg = arg
    
    def table_head(self):
        ## Display Table Head
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
        return

    def table_body(self):
        ## Display Table Body
        overall = {}
        for name, results in self.entries.items():
            final_results = []
            overall = {}
            overall['o_all'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['1'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['2'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['3'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['4'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])

            sub_final_results = []
            sub_overall = {}
            sub_overall['1'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            sub_overall['2'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            sub_overall['3'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            sub_overall['4'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])

            for dataset, (ade, fde, s, l, i, ni, lf, ca, grp, oth) in results.items():
                overall['1'] += np.array([s['N'], s['kf'][0] * s['N'], s['kf'][1] * s['N'],
                                                s['kf'][2], s['kf'][3], s['kf'][4]])
                overall['2'] += np.array([l['N'], l['kf'][0] * l['N'], l['kf'][1] * l['N'],
                                                l['kf'][2], l['kf'][3], l['kf'][4]])
                overall['3'] += np.array([i['N'], i['kf'][0] * i['N'], i['kf'][1] * i['N'],
                                                i['kf'][2], i['kf'][3], i['kf'][4]])
                overall['4'] += np.array([ni['N'], ni['kf'][0] * ni['N'], ni['kf'][1] * ni['N'],
                                                ni['kf'][2], ni['kf'][3], ni['kf'][4]])
                overall['o_all'] += np.array([ade['N'], ade['kf'] * ade['N'], fde['kf'] * fde['N'],
                                              s['kf'][2] + l['kf'][2] + i['kf'][2] + ni['kf'][2],
                                              s['kf'][3] + l['kf'][3] + i['kf'][3] + ni['kf'][3],
                                              s['kf'][4] + l['kf'][4] + i['kf'][4] + ni['kf'][4]])
                sub_overall['1'] += np.array([lf['N'], lf['kf'][0] * lf['N'], lf['kf'][1] * lf['N'],
                                                lf['kf'][2], lf['kf'][3], lf['kf'][4]])
                sub_overall['2'] += np.array([ca['N'], ca['kf'][0] * ca['N'], ca['kf'][1] * ca['N'],
                                                ca['kf'][2], ca['kf'][3], ca['kf'][4]])
                sub_overall['3'] += np.array([grp['N'], grp['kf'][0] * grp['N'], grp['kf'][1] * grp['N'],
                                                grp['kf'][2], grp['kf'][3], grp['kf'][4]])
                sub_overall['4'] += np.array([oth['N'], oth['kf'][0] * oth['N'], oth['kf'][1] * oth['N'],
                                                oth['kf'][2], oth['kf'][3], oth['kf'][4]])
            print('')

            for keys in list(overall.keys()):
                if overall[keys][0] != 0:
                    overall[keys][1] /= overall[keys][0]
                    overall[keys][2] /= overall[keys][0]
                    overall[keys][3] /= (overall[keys][0]*0.01)
                    if overall[keys][5] != 0:
                        overall[keys][4] /= (overall[keys][5]* 0.01)
                    else:
                        overall[keys][4] = -1
                    final_results += [int(overall[keys][0]), overall[keys][1], overall[keys][2], overall[keys][3], overall[keys][4]]
                else:
                    final_results += [0, 0.0, 0.0, 0.0, 0.0]

            for keys in list(sub_overall.keys()):
                if sub_overall[keys][0] != 0:
                    sub_overall[keys][1] /= sub_overall[keys][0]
                    sub_overall[keys][2] /= sub_overall[keys][0]
                    sub_overall[keys][3] /= (sub_overall[keys][0]*0.01)
                    if sub_overall[keys][5] != 0:
                        sub_overall[keys][4] /= (sub_overall[keys][5]* 0.01)
                    else:
                        sub_overall[keys][4] = -1
                    sub_final_results += [int(sub_overall[keys][0]), sub_overall[keys][1], sub_overall[keys][2], sub_overall[keys][3], sub_overall[keys][4]]
                else:
                    sub_final_results += [0, 0.0, 0.0, 0.0, 0.0]

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

            print(
                '{name:>35s} |      |                   LF.             |'
                '              CA                   |                 Grp               |                 Oth                |'
                    .format(name=''))

            print(
                '{dataset:>35s}'
                ' |     '
                ' | {sub_final_results[0]:>4}'
                ' | {sub_final_results[1]:.2f}'
                ' | {sub_final_results[2]:.2f}'
                ' | {sub_final_results[3]:.1f}'
                '  | {sub_final_results[4]: .1f}'
                ' | {sub_final_results[5]:>4}'
                ' | {sub_final_results[6]:.2f}'
                ' | {sub_final_results[7]:.2f}'
                ' | {sub_final_results[8]:.1f}'
                '  | {sub_final_results[9]:.1f}'
                ' | {sub_final_results[10]:>4}'
                ' | {sub_final_results[11]:.2f}'
                ' | {sub_final_results[12]:.2f}'
                ' | {sub_final_results[13]:.1f}'
                '  | {sub_final_results[14]:.1f}'
                ' | {sub_final_results[15]:>4}'
                ' | {sub_final_results[16]:.2f}'
                ' | {sub_final_results[17]:.2f}'
                ' | {sub_final_results[18]:.1f}'
                '  | {sub_final_results[19]:.1f} |'
                .format(dataset=name, sub_final_results=sub_final_results))
        return

    def add_entry(self, name, results):
        self.entries[name] = results

    def print_table(self):
        self.table_head()
        self.table_body()

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


        # j = -1
        # while model[j] != '/':
        #     j -= 1
        # names.append(model[j+1:].replace('.pkl', ''))

