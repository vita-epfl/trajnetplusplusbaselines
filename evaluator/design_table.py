import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import OrderedDict

import pandas as pd
from evaluator.evaluator_helpers import Categories, Sub_categories, Metrics


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
        sub_final_results = []
        ## Overall metrics ADE, FDE, ColI, ColII, Topk_ade, Topk_fde, NLL
        table_metrics = Metrics(*([0]*8))
        ## Metrics for the 4 types of trajectories and interactions
        table_categories = Categories(*[Metrics(*([0]*8)) for i in range(1,5)])
        table_sub_categories = Sub_categories(*[Metrics(*([0]*8)) for i in range(1,5)])

        for dataset, (metrics, categories, sub_categories) in results.items():
            ## Overall
            table_metrics += metrics
            
            ## Main Types
            table_categories.static_scenes += categories.static_scenes
            table_categories.linear_scenes += categories.linear_scenes
            table_categories.forced_non_linear_scenes += categories.forced_non_linear_scenes
            table_categories.non_linear_scenes += categories.non_linear_scenes

            ## Sub Types
            table_sub_categories.lf += sub_categories.lf
            table_sub_categories.ca += sub_categories.ca
            table_sub_categories.grp += sub_categories.grp
            table_sub_categories.others += sub_categories.others

        final_results += table_categories.static_scenes.avg_vals_to_list()
        final_results += table_categories.linear_scenes.avg_vals_to_list()
        final_results += table_categories.forced_non_linear_scenes.avg_vals_to_list()
        final_results += table_categories.non_linear_scenes.avg_vals_to_list()
        final_results += table_metrics.avg_vals_to_list()

        sub_final_results += table_sub_categories.lf.avg_vals_to_list()
        sub_final_results += table_sub_categories.ca.avg_vals_to_list()
        sub_final_results += table_sub_categories.grp.avg_vals_to_list()
        sub_final_results += table_sub_categories.others.avg_vals_to_list()

        self.results[name] = final_results
        self.sub_results[name] = sub_final_results
        return final_results, sub_final_results

    def add_result(self, name, final_results, sub_final_results):
        self.results[name] = final_results
        self.sub_results[name] = sub_final_results

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

    def print_table(self):
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
        fig.savefig('Results.png')
    