import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import OrderedDict
import pandas as pd

class Table(object):
    """docstring for Table"""
    def __init__(self, arg=None):
        super(Table, self).__init__()

        self.entries = {}
        self.sub_entries = {}
        self.arg = arg
        self.results = {}
        self.sub_results = {}

    def table_body(self):
        ## Display Table Body
        overall = {}
        for name, results in self.entries.items():
            final_results = []
            overall = {}
            overall['1'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['2'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['3'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['4'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            overall['o_all'] = np.array([0, 0.0, 0.0, 0.0, 0.0, 0])
            
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

            self.results[name] = final_results
            self.sub_results[name] = sub_final_results

        self.ai_crowd()
        return

    def add_entry(self, name, results):
        self.entries[name] = results

    def print_table(self):
        self.table_body()

    def ai_crowd(self):
        fde_3 = min(1.00, 2.00)
        fde = min(1.00, 2.00)
        pred_col = min(10, 8)
        gt_col = min(10, 8)
        ade = min(0.50, 0.4)
        df = pd.DataFrame(OrderedDict({
            'group': ['Baseline', 'name'],
            'FDE for type III': [fde_3, fde_3 / 2.00],
            'FDE': [fde, fde / 2.00],
            'Pred. Colision': [pred_col / 10, pred_col / 8],
            'GT Colision': [gt_col / 10, gt_col / 8],
            'ADE': [ade / 0.5, ade / 0.4]
        }))


        categories = list(df)[1:]
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(311, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], color="grey", size=7)
        plt.ylim(0, 1.1)

        # ------- PART 2: Add plots

        # Ind1
        values = df.loc[0].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="occupancy_35_cell_6")
        ax.fill(angles, values, 'b', alpha=0.1)

        # Ind2
        values = df.loc[1].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='name')
        ax.fill(angles, values, 'r', alpha=0.1) 

    # ------------------------------------------ TABLES -------------------------------------------
        # Overall Table #
        ax1 = fig.add_subplot(312)
        ax1.axis('tight')
        ax1.axis('off')

        cellText = [['', 'ADE', 'FDE', 'Col 1', 'Col 2']]
        rowLabels = ['']
        colLabels = ['N', 'Overall', 'Overall', 'Overall', 'Overall']

        for key in self.results.keys(): 
            cellText.append([self.results[key][index].__format__('.2f') for index in range(20, 25)])
            rowLabels.append(key)
        
        the_overall_table = ax1.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels,
                                      cellLoc='center',loc='center',bbox=[0.13,0.8,0.7,0.25])

        the_overall_table.auto_set_font_size(False)
        the_overall_table.set_fontsize(8)

        for (row, col), cell in the_overall_table.get_celld().items():
            if (row == 0) or (row == 1) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        # Table Real #
        ax2 = fig.add_subplot(313)
        ax2.text(0.45, 2.1, 'REAL')
        ax2.axis('tight')
        ax2.axis('off')
        cellText = [['', 'ADE', 'FDE', 'Col 1', 'Col 2', '', 'ADE', 'FDE', 'Col 1', 'Col 2', 
                     '', 'ADE', 'FDE', 'Col 1', 'Col 2', '', 'ADE', 'FDE', 'Col 1', 'Col 2' ]]
        rowLabels = ['Error']
        colLabels = ['N', 'I', 'I', 'I', 'I', 'N', 'II', 'II', 'II', 'II', 
                     'N', 'III', 'III','III', 'III', 'N', 'IV', 'IV', 'IV', 'IV']

        for key in self.results.keys(): 
            cellText.append([self.results[key][index].__format__('.2f') for index in range(20)])
            rowLabels.append(key)

        cellText.append(['N', 'LF', 'LF', 'LF', 'LF', 'N', 'CA', 'CA', 'CA', 'CA', 
                        'N', 'Grp', 'Grp','Grp', 'Grp', 'N', 'Oth', 'Oth', 'Oth', 'Oth'])
        cellText.append(['', 'ADE', 'FDE', 'Col 1', 'Col 2', '', 'ADE', 'FDE', 'Col 1', 'Col 2', 
                        '', 'ADE', 'FDE', 'Col 1', 'Col 2', '', 'ADE', 'FDE', 'Col 1', 'Col 2' ])
        rowLabels.append('')
        rowLabels.append('Error')

        for key in self.sub_results.keys(): 
            cellText.append([self.sub_results[key][index].__format__('.2f') for index in range(20)])
            rowLabels.append(key)

        the_table = ax2.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels,
                              cellLoc='center',loc='center',bbox=[0, 1.5, 1, 0.3])

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        for (row, col), cell in the_table.get_celld().items():
            if (row == 0) or (row == 1) or (col == -1) or (row == len(self.results.keys())+2) or (row == len(self.results.keys())+3):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.savefig('Radar.png', bbox_inches='tight')
        plt.show()
        plt.close()