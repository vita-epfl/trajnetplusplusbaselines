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
        self.collision_test = {}

    def add_collision_entry(self, name, result):
        self.collision_test[name] = result

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
                final_results += [int(overall[keys][0]), overall[keys][1], overall[keys][2], overall[keys][4], overall[keys][3],
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
                sub_final_results += [int(sub_overall[keys][0]), sub_overall[keys][1], sub_overall[keys][2], sub_overall[keys][4], sub_overall[keys][3], 
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

    def render_mpl_table(self, data, col_width=3.0, row_height=0.625, font_size=14,
                             header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                             bbox=[0, 0, 1, 1], header_columns=0,
                             ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

        for (row, col), cell in mpl_table.get_celld().items():
            if (row == 0) or (col == 1) or (col == 0):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        return ax

    def all_chart(self):
        fig = plt.figure(figsize=(20, 20))
    # ------------------------------------------ TABLES -------------------------------------------
        # Overall Table #
        ax1 = fig.add_subplot(311)
        ax1.axis('tight')
        ax1.axis('off')

        df = pd.DataFrame(columns=['', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL', 'Col_test'])
        it = 0
        len_name = 10
        for key in self.results:
            df.loc[it] = ['Overall'] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(32, 40)] + [self.collision_test[key]]
            it += 1
        ax1 = self.render_mpl_table(df, header_columns=0, col_width=2.0, bbox=[0, 0.9, 1, 0.1*len(self.results)], ax=ax1)



        ax2 = fig.add_subplot(312)
        ax2.axis('tight')
        ax2.axis('off')
        # Overall Table #
        df = pd.DataFrame(columns=['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL'])

        type_list = [['I', ''], ['II', ''], ['III', ''], ['III', 'LF'], ['III', 'CA'], ['III', 'Grp'], ['III', 'Oth'], ['IV', '']]
        it = 0

        ##Type I
        for key in self.results:    
            df.loc[it] = type_list[0] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(8)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type II
        for key in self.results:  
            df.loc[it] = type_list[1] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(8, 16)] 
            it += 1        

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type III
        for key in self.results:  
            df.loc[it] = type_list[2] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(16, 24)] 
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type III: LF
        for key in self.results:  
            df.loc[it] = type_list[3] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(8)] 
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type III: CA
        for key in self.results:  
            df.loc[it] = type_list[4] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(8, 16)] 
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type III: Grp
        for key in self.results:  
            df.loc[it] = type_list[5] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(16, 24)] 
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type III: Others
        for key in self.results:  
            df.loc[it] = type_list[6] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(24, 32)] 
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL']
        it += 1

        ##Type IV
        for key in self.results:  
            df.loc[it] = type_list[7] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(24, 32)] 
            it += 1


        ax2 = self.render_mpl_table(df, header_columns=0, col_width=2.0, bbox=[0, -1.6, 1, 0.6*len(self.results)], ax=ax2)

        ## SYNTH
        # ax3 = fig.add_subplot(313)
        # ax3.axis('tight')
        # ax3.axis('off')
        # # Overall Table #
        # df = pd.DataFrame(columns=['Type', 'Sub-Type', 'No.', 'ADE', 'FDE', 'Col I', 'Col II', 'Top3 ADE', 'Top3 FDE', 'NLL'])

        # type_list = [['I', ''], ['II', ''], ['III', ''], ['III', 'LF'], ['III', 'CA'], ['III', 'Grp'], ['III', 'Oth'], ['IV', '']]
        # for key in self.results: 
        #     print(self.sub_results[key])    
        #     print(self.results[key])    
        #     df.loc[0] = type_list[0] + [self.results[key][index].__format__('.2f') for index in range(8)]
        #     df.loc[1] = type_list[1] + [self.results[key][index].__format__('.2f') for index in range(8, 16)] 
        #     df.loc[2] = type_list[2] + [self.results[key][index].__format__('.2f') for index in range(16, 24)] 
        #     df.loc[3] = type_list[3] + [self.sub_results[key][index].__format__('.2f') for index in range(8)] 
        #     df.loc[4] = type_list[4] + [self.sub_results[key][index].__format__('.2f') for index in range(8, 16)] 
        #     df.loc[5] = type_list[5] + [self.sub_results[key][index].__format__('.2f') for index in range(16, 24)] 
        #     df.loc[6] = type_list[6] + [self.sub_results[key][index].__format__('.2f') for index in range(24, 32)] 
        #     df.loc[7] = type_list[7] + [self.results[key][index].__format__('.2f') for index in range(24, 32)] 


        # ax3 = self.render_mpl_table(df, header_columns=0, col_width=2.0, ax=ax3)

        # fig = ax.get_figure()
        fig.savefig('Results.png')
    