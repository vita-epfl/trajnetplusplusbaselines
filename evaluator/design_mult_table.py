import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import OrderedDict

class Table(object):
    """docstring for Table"""
    def __init__(self, arg=None):
        super(Table, self).__init__()

        self.entries = {}
        self.sub_entries = {}
        self.arg = arg
        self.results = {}
        self.sub_results = {}

    def add_entry(self, name, results):
        final_results = []
        overall = {}
        overall['1'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        overall['2'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        overall['3'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        overall['4'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        overall['o_all'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        
        sub_final_results = []
        sub_overall = {}
        sub_overall['1'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        sub_overall['2'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        sub_overall['3'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
        sub_overall['4'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])

        for dataset, (ade, fde, s, l, i, ni, lf, ca, grp, oth, k_ade, k_fde, nll) in results.items():
            ## Overall
            overall['o_all'] += np.array([ade['N'], ade['kf'] * ade['N'], fde['kf'] * fde['N'],
                                          s['kf'][2] + l['kf'][2] + i['kf'][2] + ni['kf'][2],
                                          s['kf'][3] + l['kf'][3] + i['kf'][3] + ni['kf'][3],
                                          s['kf'][4] + l['kf'][4] + i['kf'][4] + ni['kf'][4],
                                          k_ade['kf'] * k_ade['N'], k_fde['kf'] * k_fde['N'], nll['kf'] * nll['N']])

            ## Static
            overall['1'] += np.array([s['N'], s['kf'][0] * s['N'], s['kf'][1] * s['N'],
                                            s['kf'][2], s['kf'][3], s['kf'][4], s['kf'][5] * s['N'], s['kf'][6] * s['N'], s['kf'][7] * s['N']])
            ## Linear
            overall['2'] += np.array([l['N'], l['kf'][0] * l['N'], l['kf'][1] * l['N'],
                                            l['kf'][2], l['kf'][3], l['kf'][4], l['kf'][5] * l['N'], l['kf'][6] * l['N'], l['kf'][7] * l['N']])
            ## Interacting
            overall['3'] += np.array([i['N'], i['kf'][0] * i['N'], i['kf'][1] * i['N'],
                                            i['kf'][2], i['kf'][3], i['kf'][4], i['kf'][5] * i['N'], i['kf'][6] * i['N'], i['kf'][7] * i['N']])
            ## Non-Interacting
            overall['4'] += np.array([ni['N'], ni['kf'][0] * ni['N'], ni['kf'][1] * ni['N'],
                                            ni['kf'][2], ni['kf'][3], ni['kf'][4], ni['kf'][5] * ni['N'], ni['kf'][6] * ni['N'], ni['kf'][7] * ni['N']])

            ## Leader Follower
            sub_overall['1'] += np.array([lf['N'], lf['kf'][0] * lf['N'], lf['kf'][1] * lf['N'],
                                            lf['kf'][2], lf['kf'][3], lf['kf'][4], lf['kf'][5] * lf['N'], lf['kf'][6] * lf['N'], lf['kf'][7] * lf['N']])
            ## Collision Avoidance
            sub_overall['2'] += np.array([ca['N'], ca['kf'][0] * ca['N'], ca['kf'][1] * ca['N'],
                                            ca['kf'][2], ca['kf'][3], ca['kf'][4], ca['kf'][5] * ca['N'], ca['kf'][6] * ca['N'], ca['kf'][7] * ca['N']])
            ## Group
            sub_overall['3'] += np.array([grp['N'], grp['kf'][0] * grp['N'], grp['kf'][1] * grp['N'],
                                            grp['kf'][2], grp['kf'][3], grp['kf'][4], grp['kf'][5] * grp['N'], grp['kf'][6] * grp['N'], grp['kf'][7] * grp['N']])
            ## Others
            sub_overall['4'] += np.array([oth['N'], oth['kf'][0] * oth['N'], oth['kf'][1] * oth['N'],
                                            oth['kf'][2], oth['kf'][3], oth['kf'][4], oth['kf'][5] * oth['N'], oth['kf'][6] * oth['N'], oth['kf'][7] * oth['N']])
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
                overall[keys][6] /= overall[keys][0]
                overall[keys][7] /= overall[keys][0]
                overall[keys][8] /= overall[keys][0]
                final_results += [int(overall[keys][0]), overall[keys][1], overall[keys][2], overall[keys][3], overall[keys][4],
                                  overall[keys][6], overall[keys][7], overall[keys][8]]
            else:
                final_results += [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        for keys in list(sub_overall.keys()):
            if sub_overall[keys][0] != 0:
                sub_overall[keys][1] /= sub_overall[keys][0]
                sub_overall[keys][2] /= sub_overall[keys][0]
                sub_overall[keys][3] /= (sub_overall[keys][0]*0.01)
                if sub_overall[keys][5] != 0:
                    sub_overall[keys][4] /= (sub_overall[keys][5]* 0.01)
                else:
                    sub_overall[keys][4] = -1
                sub_overall[keys][6] /= sub_overall[keys][0]
                sub_overall[keys][7] /= sub_overall[keys][0]
                sub_overall[keys][8] /= sub_overall[keys][0]
                sub_final_results += [int(sub_overall[keys][0]), sub_overall[keys][1], sub_overall[keys][2], sub_overall[keys][3], sub_overall[keys][4], 
                                      sub_overall[keys][6], sub_overall[keys][7], sub_overall[keys][8]]
            else:
                sub_final_results += [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.results[name] = final_results
        self.sub_results[name] = sub_final_results
        return final_results, sub_final_results

    def add_result(self, name, final_results, sub_final_results):
        self.results[name] = final_results
        self.sub_results[name] = sub_final_results

    def print_table(self):
        self.all_chart()

    def all_chart(self):
        fig = plt.figure(figsize=(20, 20))
    # ------------------------------------------ TABLES -------------------------------------------
        # Overall Table #
        ax1 = fig.add_subplot(211)
        ax1.axis('tight')
        ax1.axis('off')

        cellText = [['', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll']]
        rowLabels = ['']
        colLabels = ['N', 'Overall', 'Overall', 'Overall', 'Overall', 'Overall', 'Overall', 'Overall']

        for key in self.results.keys(): 
            cellText.append([self.results[key][index].__format__('.2f') for index in range(32, 40)])
            rowLabels.append(key)
        
        the_overall_table = ax1.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels,
                                      cellLoc='center',loc='center',bbox=[0.13,0.8,0.7,0.25])

        the_overall_table.auto_set_font_size(False)
        the_overall_table.set_fontsize(8)

        for (row, col), cell in the_overall_table.get_celld().items():
            if (row == 0) or (row == 1) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        # Table Real #
        ax2 = fig.add_subplot(212)
        ax2.axis('tight')
        ax2.axis('off')
        cellText = [['', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll', '', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll', 
                     '', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll', '', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll' ]]
        rowLabels = ['Error']
        colLabels = ['N', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'N', 'II', 'II', 'II', 'II', 'II', 'II', 'II', 
                     'N', 'III', 'III','III', 'III', 'III','III', 'III', 'N', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV']

        for key in self.results.keys(): 
            cellText.append([self.results[key][index].__format__('.2f') for index in range(32)])
            rowLabels.append(key)

        cellText.append(['N', 'LF', 'LF', 'LF', 'LF', 'LF', 'LF', 'LF', 'N', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 
                        'N', 'Grp', 'Grp','Grp', 'Grp', 'Grp','Grp', 'Grp', 'N', 'Oth', 'Oth', 'Oth', 'Oth', 'Oth', 'Oth', 'Oth'])
        cellText.append(['', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll', '', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll', 
                        '', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll', '', 'ADE', 'FDE', 'Col 1', 'Col 2', 'kADE', 'kFDE', 'nll' ])
        rowLabels.append('')
        rowLabels.append('Error')

        for key in self.sub_results.keys(): 
            cellText.append([self.sub_results[key][index].__format__('.2f') for index in range(32)])
            rowLabels.append(key)

        the_table = ax2.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels,
                              cellLoc='center',loc='center',bbox=[0, 1.5, 1, 0.3])

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        for (row, col), cell in the_table.get_celld().items():
            if (row == 0) or (row == 1) or (col == -1) or (row == len(self.results.keys())+2) or (row == len(self.results.keys())+3):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.savefig('Table.png', bbox_inches='tight')
        plt.show()
        plt.close()