# Setup
import math
import joblib
from utils import *
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.stats import ttest_rel


# Define directories path:
TAG = ''  # '_low_BMI', '_high_AHIvariability', '_low_AHIvariability'
MODEL_TYPE = 'LR_lasso'
USE_XGB_FOR_MEDICATION = False
DATASET_TO_NAME = load_dataset_to_name()
DICT_FOR_BARPLOT = DATASET_TO_NAME.copy()
DICT_FOR_BARPLOT['MBpathways'] = f'Gut MB metabolic\npathways'
DICT_FOR_BARPLOT['cardiovascular'] = 'Cardiovascular\nsystem'
NO_OF_DS_SELECTION = 5


def plot_features_importance(path: str,
                             model_name: str,
                             model_file: str,
                             dataset_name: str):
    """ Plot the contribution level of the most important features in the model fitting a specific targe
    The model name should be one of the following:
     'LR' (for regular linear regression), 'LR_lasso', 'LR_ridge', 'LGBM' or 'XGB'
    """
    filename = os.path.join(path, model_file)
    model = joblib.load(f'{filename}.sav')
    if model_name == 'LR' or model_name == 'LR_lasso':
        f_importances = pd.Series(model.coef_, index=model.feature_names_in_)
    else:  # elif model_name == 'LGBM':
        f_importances = pd.Series(model.feature_importances_, index=model.feature_name_)
    sorted_data = f_importances.abs().sort_values(ascending=True)
    plt.figure(figsize=(10, 10), constrained_layout=True)
    sorted_data[-10:].plot.barh()
    plt.yticks(rotation=45)
    plt.title(f"{dataset_name}\nFeature Importances in {model_name} fitting")
    fig_dir = mkdirifnotexists(os.path.join(path, f'plots_{model_name}', 'feature_importances'))
    plt.savefig(os.path.join(fig_dir, f'{model_file}_feature_importances'), dpi=200)
    plt.close()  # show()
    pass


def plot_regression_scores_heatmap(name: str,
                                   scores: pd.DataFrame,
                                   target_list: list,
                                   path_for_fig: str,
                                   task: str,
                                   special_annot: pd.DataFrame = None,
                                   scale: tuple = None):
    """ Heatmap with the prediction performance of a model based on
    Age, Sex and BMI VS. same with addition of a dataset """
    if scale is None:
        scale = (scores.min().min(), scores.max().max())
    if special_annot is None:
        annot = True
        fmt = ".2f"
    else:
        annot = special_annot.loc[target_list]  # special_annot.transpose()[target_list]
        fmt = ""
    for col_name in list(scores):
        if col_name != 'Age & BMI':
            scores.rename(columns={col_name: 'Age, BMI + \n' + col_name}, inplace=True)
    plt.figure(constrained_layout=True, figsize=(8, 8))
    sns.heatmap(data=scores.loc[target_list],  # scores.transpose()[target_list],
                annot=annot,
                fmt=fmt,
                vmin=scale[0],
                vmax=scale[1],
                cmap="Blues",
                linewidths=.75,
                )
    plt.yticks(rotation=0)
    sample_sizes = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(path_for_fig)), 'sample_sizes_full.csv'),
                               index_col=0, header=0).squeeze('columns')
    if 'sleep_quality_best_night' in task:
        new_index_labels = [index.replace('sleep_quality_avg', 'sleep_quality_best_night') for index in
                            sample_sizes.index]
        sample_sizes.index = new_index_labels
    head, tail = os.path.split(path_for_fig)
    head, tail = os.path.split(head)
    sample_size = sample_sizes.loc[f'{tail}_{name}']
    plt.title(f'{tail}_{name} N={sample_size}\nPearson corr btw predicted and actual values')
    plt.savefig(os.path.join(path_for_fig, f'{name}_Regressions_score_TTest_rel'), dpi=200)
    plt.close()
    pass


def find_significant_predictions(name: str, path: str, file: str, datasets_dict: dict, target_group: list, task: str):
    """ Function that runs over a regression score file, detects the features significantly predicted:
     by running a T-Test between the 2 groups prediction scores (model based on baseline VS baseline + dataset)
     and plot the results into a condensed heatmap"""
    # Create a dataframe with all median scores
    median_scores = pd.concat([pd.read_csv(os.path.join(path, f'{df_prefix}_{file}'), index_col=0
                                           ).median(axis=0) for df_prefix in datasets_dict.keys()], axis=1)
    median_scores.columns = datasets_dict.values()
    # Create directory with the results
    if 'medications' in path and USE_XGB_FOR_MEDICATION:
        model_type = 'XGB'
    else:
        model_type = MODEL_TYPE
    fig_dir = mkdirifnotexists(os.path.join(path, f'plots_{model_type}'))
    median_scores.to_csv(os.path.join(fig_dir, f'median_scores-{name}_dataset.csv'))
    median_scores = median_scores.dropna()
    if target_group is None:
        target_group = median_scores.index.to_list()
    mask = pd.DataFrame(index=target_group, columns=median_scores.columns)
    for target in target_group:
        scores_per_target = pd.concat([pd.read_csv(os.path.join(path, f'{df_prefix}_{file}'),
                                                   index_col=0)[target] for df_prefix in datasets_dict.keys()],
                                      axis=1)
        scores_per_target.columns = datasets_dict.values()
        if 0:  # Boxplot of the scores distribution per target:
            tmp = scores_per_target.copy()
            for col_name in list(tmp):
                if col_name != 'Age & BMI':
                    tmp.rename(columns={col_name: 'Age, BMI + ' + col_name}, inplace=True)
            plt.figure(constrained_layout=True)
            sns.boxplot(data=tmp, palette='Set2', orient='h')
            plt.axvline(x=median_scores.loc[target, 'Age & BMI'], color='black', linestyle='--')
            plt.xlabel('pearson correlation prediction vs actual')
            plt.title(f'Ability to predict {target} - {name} dataset')
            plt.savefig(os.path.join(fig_dir, f'Regressions_scores_for_{target}-{name}_dataset.png'), dpi=200)
            plt.close()  # plt.show()
        # Perform a t-test to the body systems with significantly better results:
        for i in scores_per_target.columns:
            if i == 'Age & BMI':
                mask.loc[target, i] = (median_scores.loc[target, i] < 0)
            else:
                t_stat, p_val = ttest_rel(scores_per_target['Age & BMI'], scores_per_target[i])
                mask.loc[target, i] = (p_val > 0.001) or \
                                      (median_scores.loc[target, i] < median_scores.loc[target, 'Age & BMI']) or \
                                      (median_scores.loc[target, i] < 0) or \
                                      (median_scores.loc[target, 'Age & BMI'] < 0)
    mask.to_csv(os.path.join(fig_dir, f'mask-{name}_dataset.csv'))
    # Remove the masked rows and reorder:
    median_scores = median_scores.mask(mask)
    median_scores.dropna(how='any', axis=0, inplace=True)
    median_scores.fillna(0, inplace=True)
    if median_scores.shape[0] > 2:
        median_scores['diff'] = median_scores.iloc[:, 1] - median_scores.iloc[:, 0]
        median_scores.sort_values(by='diff', ascending=False, inplace=True)
        median_scores.drop(['diff'], axis=1, inplace=True)
        # median_scores = reorder_by_clusters(median_scores)
    mask = mask.loc[median_scores.index, median_scores.columns]
    target_group = [element for element in median_scores.index if element in target_group]
    median_scores.to_csv(os.path.join(fig_dir, f'median_scores_filtered-{name}_dataset.csv'))
    # Heatmap of the regression models scores (median):
    if median_scores.shape[0] == 0:
        print(f'No significant association was found for this dataset - {name}')
    else:
        plot_regression_scores_heatmap(name=name, scores=median_scores, target_list=target_group,
                                       path_for_fig=fig_dir, task=task, scale=(0, 1))
    pass


def plot_figures_for_paper(body_systems: list, task_path_dict: dict):
    """ Script to create the figures in the paper related to the regression results
     - Boxplots condensing all significant regression results
     - Performance of specific target predictions
     """
    for task in task_path_dict.keys():
        no_of_features = pd.DataFrame(index=body_systems,
                                      columns=['no_of_features_significantly_predicted', 'total_features'])
        for sex in ['men', 'women']:
            # Create datasets for plots:
            predictive_power_dataset = {}
            diff_dataset = {}
            for name in body_systems:
                if name == 'medications' and USE_XGB_FOR_MEDICATION:
                    model_type = 'XGB'
                else:
                    model_type = MODEL_TYPE
                if 'from' in task:
                    from_dataset = task[len('from_'):]
                    predict_dataset = name
                elif 'predict' in task:
                    from_dataset = name
                    predict_dataset = task[len('predict_'):]
                dir_path = os.path.join(task_path_dict[task], f'{from_dataset}_and_{predict_dataset}', f'plots_{model_type}')
                median_scores_filtered = pd.read_csv(os.path.join(dir_path, f'median_scores_filtered-{sex}_dataset.csv'),
                                                     index_col=0)
                no_of_features.loc[name, 'total_features'] = pd.read_csv(os.path.join(
                    dir_path, f'median_scores-{sex}_dataset.csv'), index_col=0).shape[0]
                no_of_features.loc[name, 'no_of_features_significantly_predicted'] = median_scores_filtered.shape[0]
                predictive_power_dataset[name] = median_scores_filtered.iloc[:, 1].values
                diff_dataset[name] = median_scores_filtered.iloc[:, 1].values - median_scores_filtered.iloc[:, 0].values
            sorted_categories = sorted(diff_dataset.keys(), key=lambda x: (
                float('-inf') if pd.isna(np.median(diff_dataset[x])) else np.median(diff_dataset[x])), reverse=True)

            label_fontsize = 20
            tick_fontsize = 15
            ticks_interval = 0.05
            markers_size = 4
            palette = 'tab10'
            colors = plt.get_cmap('tab20')
            custom_palette = {col: sns.color_palette([colors((idx % 10) * 2 + 1), colors((idx % 10) * 2)]) for idx, col in
                              enumerate(body_systems)}
            fig = plt.figure(figsize=(14, 12))  # 16, 15))
            gs = GridSpec(2 * NO_OF_DS_SELECTION, 12, wspace=10., hspace=2.)

            # Figure A:
            # Plot boxplots with the difference btw predictive power of each dataset vs baseline
            top_boxplots = fig.add_subplot(gs[:NO_OF_DS_SELECTION - 1, :6])
            df = pd.DataFrame({key: pd.Series(value) for key, value in diff_dataset.items()})
            # sns.violinplot(data=df, order=sorted_categories, palette=palette, orient='v', ax=top_boxplots)
            sns.swarmplot(data=df, order=sorted_categories, palette=palette, orient='v', size=markers_size,
                          ax=top_boxplots)
            sns.boxplot(data=df, order=sorted_categories, orient='v', color='black', fill=False, showfliers=False,
                        ax=top_boxplots)
            no_of_features = no_of_features.reindex(sorted_categories)
            top_boxplots.set_xticklabels(
                [f'({no_of_features.iloc[r, 0]})/({no_of_features.iloc[r, 1]})' for r in range(len(no_of_features.index))],
                fontsize=tick_fontsize, rotation=45, ha='right')
            top_boxplots.tick_params(axis='y', which='both', left=False, right=False, labelsize=tick_fontsize)
            top_boxplots.yaxis.set_major_locator(MultipleLocator(ticks_interval))
            top_boxplots.set_ylabel('Predictive power difference\nfrom baseline model\n', fontsize=tick_fontsize)
            # Plot boxplots with the predictive power of each dataset:
            bottom_boxplots = fig.add_subplot(gs[NO_OF_DS_SELECTION:-1, :6])
            df = pd.DataFrame({key: pd.Series(value) for key, value in predictive_power_dataset.items()})
            filtered_column_map = {key: value for key, value in DATASET_TO_NAME.items() if key in df.columns}
            df.rename(columns=filtered_column_map, inplace=True)
            renamed_sorted_categories = [DATASET_TO_NAME[cat] for cat in sorted_categories]
            # sns.violinplot(data=df, order=renamed_sorted_categories, palette=palette, orient='v',
            #               ax=bottom_boxplots)
            sns.swarmplot(data=df, order=renamed_sorted_categories, palette=palette, orient='v', size=markers_size,
                          ax=bottom_boxplots)
            sns.boxplot(data=df, order=renamed_sorted_categories, orient='v', color='black', fill=False,
                        showfliers=False, ax=bottom_boxplots)
            bottom_boxplots.set_xticklabels(bottom_boxplots.get_xticklabels(),
                                            fontsize=tick_fontsize, rotation=45, ha='right')
            bottom_boxplots.tick_params(axis='y', which='both', left=False, right=False, labelsize=tick_fontsize)
            bottom_boxplots.yaxis.set_major_locator(MultipleLocator(2 * ticks_interval))
            bottom_boxplots.set_ylabel(f'Predictive power\n',  # (Pearson R between predicted and actual values)\n',
                                       fontsize=tick_fontsize)
            # if sex == 'men':
            #     bottom_boxplots.set_ylim(0, 0.7)

            # Figure B: Deep dive into the performance of the most significant predicted target for each of the following datasets
            picked_datasets = sorted_categories[:NO_OF_DS_SELECTION]
            axs = [0] * NO_OF_DS_SELECTION
            x_max = 0.0
            mbfamily_names = mbfamily_to_name()
            for i, picked_dataset in enumerate(picked_datasets):
                axs[i] = fig.add_subplot(gs[2 * i:2 * i + 2, 8:])
                if picked_dataset == 'medications' and USE_XGB_FOR_MEDICATION:
                    model_type = 'XGB'
                else:
                    model_type = MODEL_TYPE
                if 'from' in task:
                    predict_dataset = picked_dataset
                else:  # 'predict' in task
                    from_dataset = picked_dataset
                path = os.path.join(task_path_dict[task], f'{from_dataset}_and_{predict_dataset}')
                df = pd.read_csv(os.path.join(path, f'plots_{model_type}', f'median_scores_filtered-{sex}_dataset.csv'),
                                 index_col=0)
                if df.empty:
                    print(f'No significant predictions found from dataset: {picked_dataset}')
                    continue
                else:
                    df['diff'] = df.iloc[:, 1] - df.iloc[:, 0]
                    target = df.sort_values(by='diff', ascending=False).index[0]

                tmp_dict = {f'Age_Gender_BMI_and_{predict_dataset}': DICT_FOR_BARPLOT['Age_Gender_BMI'],
                            f'{from_dataset}_and_{predict_dataset}': DICT_FOR_BARPLOT[from_dataset]}
                file = f'{model_type}_{sex}scores_df.csv'
                scores_per_target = pd.concat([pd.read_csv(os.path.join(path, f'{df_prefix}_{file}'),
                                                           index_col=0)[target] for df_prefix in tmp_dict.keys()],
                                              axis=1)
                scores_per_target.columns = tmp_dict.values()
                tmp = scores_per_target.copy()
                for col_name in list(tmp):
                    if col_name != 'Age & BMI':
                        tmp.rename(columns={col_name: f'Age, BMI \n& {col_name}'}, inplace=True)
                sns.barplot(tmp, orient='h', errorbar='sd', palette=custom_palette[picked_dataset], ax=axs[i])
                sns.despine()
                axs[i].bar_label(axs[i].containers[0], fmt='%.3f', color=custom_palette[picked_dataset][0],
                                 fontsize=tick_fontsize - 2, padding=3)
                axs[i].bar_label(axs[i].containers[1], fmt='%.3f', color=custom_palette[picked_dataset][1],
                                 fontsize=tick_fontsize - 2, padding=3)
                axs[i].tick_params(axis='y', which='both', left=False, right=False, labelsize=tick_fontsize)
                axs[i].tick_params(axis='x', which='both', top=False, bottom=False, labelsize=tick_fontsize)
                if (picked_dataset == 'diet') and ('from' in task):
                    target = f'{target} intake'
                elif picked_dataset == 'blood_lipids' and ('from' in task):
                    target = f'{target[:15]}*'
                elif picked_dataset == 'MBfamily2' and ('fBin' in target):
                    target = f'MB family: {mbfamily_names[target]}'
                elif picked_dataset == 'sleep_quality_avg' or ('predict_sleep' in task):
                    target = target.replace('ahi', 'AHI')
                    target = target.replace('odi', 'ODI')
                    target = target.replace('rdi', 'RDI')
                target = target.replace('_', ' ')
                target = target.replace('bt ', 'BT')
                target = target.replace('rds', 'RDS')
                axs[i].set_title(f'{target}', fontsize=tick_fontsize)
                if np.floor(i) == np.floor(len(picked_datasets) / 2):
                    axs[i].set_ylabel('Model based on', fontsize=label_fontsize)

                curr_x_max = tmp.max().max() + 0.1
                if curr_x_max > x_max:
                    x_max = curr_x_max

            ticks_interval = math.floor(x_max * 10)
            if ticks_interval % 2:
                ticks_interval /= 10
            else:
                ticks_interval /= 20
            for i in range(0, NO_OF_DS_SELECTION):
                axs[i].set_xlim(0, x_max)
                axs[i].xaxis.set_major_locator(MultipleLocator(ticks_interval))
                axs[i].set_xticklabels(axs[i].get_xticklabels(), fontsize=tick_fontsize)
                if i == NO_OF_DS_SELECTION - 1:
                    axs[i].set_xlabel(f'Predictive power', fontsize=tick_fontsize)
                else:
                    axs[i].get_xaxis().set_visible(False)

            plt.tight_layout()
            if USE_XGB_FOR_MEDICATION:
                plt.savefig(os.path.join(task_path_dict[task], f'Fig-{task}_{sex}.png'), dpi=200)
            else:
                plt.savefig(os.path.join(task_path_dict[task], f'Fig-{task}_{sex}_{MODEL_TYPE}.png'), dpi=200)
            plt.show()
            print(f'Figure saved in {task_path_dict[task]}')
    pass


def plot_age_bmi_predictions(tasks: list):
    """ Script to create the figures in the paper related to the regression results:
     - Performance of Age and BMI predictions, using Sleep quality vs HRV based models
     """
    target_names = {'age': 'Age', 'bmi': 'BMI'}
    tick_fontsize = 12
    ticks_interval = 0.3

    # First option for plot
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    for i, target in enumerate(target_names.keys()):
        vmax = 0.0
        for s, sex in enumerate(['men', 'women']):
            ax = axs[s, i]
            file = f'{MODEL_TYPE}_{sex}scores_df.csv'
            tmp_dict = {f'{ds}_and_Age_Gender_BMI': DICT_FOR_BARPLOT[ds] for ds in tasks}
            path = [os.path.join(MY_DIR, 'body_systems_associations', f'from_{ds}', 'regressions_results' + TAG,
                                 f'{ds}_and_Age_Gender_BMI') for ds in tasks]
            scores_per_target = pd.concat([pd.read_csv(os.path.join(p, f'{df_prefix}_{file}'),
                                                       index_col=0)[target] for p, df_prefix in
                                           zip(path, tmp_dict.keys())],
                                          axis=1)
            scores_per_target.columns = tmp_dict.values()
            tmp = scores_per_target.copy()
            custom_palette = sns.color_palette('Paired')
            if s == 1: #women
                custom_palette = custom_palette[4:6]
            sns.barplot(tmp, orient='v', errorbar='sd', palette=custom_palette, ax=ax)
            sns.despine()
            ax.bar_label(ax.containers[0], fmt='%.3f', color=plt.get_cmap('Paired').colors[4 * s + 0],
                         fontsize=tick_fontsize - 2, padding=3)
            ax.bar_label(ax.containers[1], fmt='%.3f', color=plt.get_cmap('Paired').colors[4 * s + 1],
                         fontsize=tick_fontsize - 2, padding=3)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelsize=tick_fontsize)
            ax.tick_params(axis='x', which='both', top=False, bottom=False, labelsize=tick_fontsize)
            if s == 0: #men
                ax.set_title(f'{target_names[target]}', fontsize=tick_fontsize)
            if i == 1:
                ax.set_yticklabels('')
            else:
                ax.set_ylabel(f'{sex}', fontsize=tick_fontsize + 2)
            curr_max = tmp.max().max() + 0.1
            if curr_max > vmax:
                vmax = curr_max

        for k in range(0, 2):
            axs[k, i].set_ylim(0, vmax)
            axs[k, i].yaxis.set_major_locator(MultipleLocator(ticks_interval))
            axs[k, i].set_yticklabels(axs[k, i].get_yticklabels(), fontsize=tick_fontsize)
            if k == 0:
                axs[k, i].get_xaxis().set_visible(False)
            else:
                axs[k, i].set_xticklabels(axs[k, i].get_xticklabels(), rotation=45, ha='right', fontsize=tick_fontsize)

    axs[1, 0].set_ylabel(f'Predictive power\nfor female', fontsize=tick_fontsize)
    axs[0, 0].set_ylabel(f'Predictive power\nfor male', fontsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(MY_DIR, 'descriptive_data_and_figures', 'Fig-Age_BMI_predictions.png'), dpi=200)
    plt.show()

    # Second option for plot
    custom_palette = {'male': MALE_COLOR, #sns.color_palette('Paired')[0],
                      'female': FEMALE_COLOR} #sns.color_palette('Paired')[4]}
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    legend_flag = False
    for i, target in enumerate(target_names.keys()):
        ax = axs[i]
        vmax = 0.0
        scores = pd.DataFrame()
        for s, sex in enumerate(['men', 'women']):
            file = f'{MODEL_TYPE}_{sex}scores_df.csv'
            tmp_dict = {f'{ds}_and_Age_Gender_BMI': DICT_FOR_BARPLOT[ds] for ds in tasks}
            path = [os.path.join(MY_DIR, 'body_systems_associations', f'from_{ds}', 'regressions_results' + TAG,
                                 f'{ds}_and_Age_Gender_BMI') for ds in tasks]
            scores_per_target = pd.concat([pd.read_csv(os.path.join(p, f'{df_prefix}_{file}'),
                                                       index_col=0)[target] for p, df_prefix in
                                           zip(path, tmp_dict.keys())],
                                          axis=1)
            scores_per_target.columns = tmp_dict.values()
            tmp = scores_per_target.copy()
            if sex == 'men':
                scores_per_target['sex'] = 'male'
            else:
                scores_per_target['sex'] = 'female'
            scores = pd.concat([scores, scores_per_target], axis=0)
            curr_max = tmp.max().max() + 0.1
            if curr_max > vmax:
                vmax = curr_max

        melted_df = scores.melt(id_vars='sex', var_name='model', value_name='score')
        if i:
            legend_flag = True
        sns.barplot(melted_df, x='model', y='score', hue='sex', orient='v', errorbar='sd', palette=custom_palette,
                    legend=legend_flag, ax=ax)
        # sns.swarmplot(melted_df, x='model', y='score', hue='sex', orient='v', palette=custom_palette,
        #             legend=legend_flag, ax=ax)
        # sns.boxplot(melted_df, x='model', y='score', hue='sex', orient='v', palette=custom_palette,
        #             legend=legend_flag, ax=ax)
        sns.despine()
        ax.bar_label(ax.containers[0], fmt='%.3f', color=custom_palette['male'],
                     fontsize=tick_fontsize - 2, padding=3)
        ax.bar_label(ax.containers[1], fmt='%.3f', color=custom_palette['female'],
                     fontsize=tick_fontsize - 2, padding=3)
        ax.set_ylim(0.25, vmax)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)
        ax.set_xlabel('')
        ax.set_ylabel('Predictive power', fontsize=tick_fontsize)
        xtick_labels = [text.get_text().split('(')[0].strip() for text in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels, #rotation=45, ha='right',
                           fontsize=tick_fontsize)
        ax.set_title(f'{target_names[target]}', fontsize=tick_fontsize)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_yticklabels('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(MY_DIR, 'descriptive_data_and_figures', f'Fig-Age_BMI_predictions2_barplot.png'), dpi=200)
    plt.show()
    a=1

    pass


if __name__ == '__main__':

    # Explore data associations with:
    target_group = None
    datasets = [
        'hematopoietic',
        'immune_system',
        'glycemic_status',
        'lifestyle',
        'mental',
        'frailty',
        'liver',
        'renal_function',
        'cardiovascular',
        'body_composition',
        'bone_density',
        'blood_lipids',
        'MBfamily2',
        'MBpathways',
        'diet',
        'medications'
    ]
    tasks = [
        'from_sleep_quality_avg',
        'from_hrv_avg',
        'predict_sleep_quality_avg'
    ]
    task_path_dict = {dir_name: os.path.join(MY_DIR, 'body_systems_associations', dir_name, 'regressions_results' + TAG)
                            for dir_name in tasks}

    if not DEMO:
        sethandlers()
        os.chdir(mkdirifnotexists(os.path.join(MY_DIR, 'Logs')))
        with qp(jobname="graphs") as q:
            q.startpermanentrun()

            for task in tasks:
                # Overall Analysis & Graphs per dataset:
                print('>>> Regression score distribution Vs baseline')
                suffixes = ['men', 'women']
                ticket_list = []
                for dataset in datasets:
                    if dataset == 'medications' and USE_XGB_FOR_MEDICATION:
                        model_type = 'XGB'
                    else:
                        model_type = MODEL_TYPE
                    model_score_files = {suffix: f'{model_type}_{suffix}scores_df.csv' for suffix in suffixes}
                    if 'from' in task:
                        from_dataset = task[len('from_'):]
                        predict_dataset = dataset
                        target_group = None
                    elif 'predict' in task:
                        from_dataset = dataset
                        predict_dataset = task[len('predict_'):]
                    else:
                        raise ValueError('Please define the features and target datasets')
                    if predict_dataset != 'Age_Gender_BMI' and from_dataset != 'Age_Gender_BMI' and \
                            predict_dataset != from_dataset:
                        print(f'{from_dataset}_and_{predict_dataset}')
                        dir_path = os.path.join(task_path_dict[task], f'{from_dataset}_and_{predict_dataset}')
                        tmp_dict = {f'Age_Gender_BMI_and_{predict_dataset}': DATASET_TO_NAME['Age_Gender_BMI'],
                                    f'{from_dataset}_and_{predict_dataset}': DATASET_TO_NAME[from_dataset]}
                        ticket_list += [q.method(find_significant_predictions,
                                                 (name, dir_path, file, tmp_dict, target_group, task))
                                        for name, file in model_score_files.items()]
    
                q.wait(ticket_list)
                print(f'<< Done processing: {task}')
    
    # plot_figures_for_paper(datasets, task_path_dict)
    plot_age_bmi_predictions(['sleep_quality_avg', 'hrv_avg'])
    print('<<< Done')
