# Setup
from utils import *
from plot_regression_results import find_significant_predictions


def create_figure_for_paper(path: str, target: str):
    entries = os.listdir(path)
    dir_list = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    all_results_men = pd.DataFrame()
    all_results_women = pd.DataFrame()
    diffs_men = pd.DataFrame()
    diffs_women = pd.DataFrame()
    for dir in dir_list:
        tmp = pd.read_csv(os.path.join(path, dir, 'regressions_results',
                                       dir.replace('from_', '') + f'_and_{target}',
                                       f'plots_LR_lasso/median_scores_filtered-men_dataset.csv'), index_col=0)
        diff = tmp.iloc[:, 1] - tmp.iloc[:, 0]
        diffs_men = pd.concat([diffs_men, diff], axis=1)
        men_result = tmp.iloc[:, 1]
        all_results_men = pd.concat([all_results_men, men_result], axis=1)
        tmp = pd.read_csv(os.path.join(path, dir, 'regressions_results',
                                       dir.replace('from_', '') + f'_and_{target}',
                                       f'plots_LR_lasso/median_scores_filtered-women_dataset.csv'), index_col=0)
        diff = tmp.iloc[:, 1] - tmp.iloc[:, 0]
        diffs_women = pd.concat([diffs_women, diff], axis=1)
        women_result = tmp.iloc[:, 1]
        all_results_women = pd.concat([all_results_women, women_result], axis=1)
    diffs_men['max'] = diffs_men.max(axis=1)
    all_results_men = pd.concat([all_results_men, diffs_men['max']], axis=1).sort_values(by='max', ascending=False)
    all_results_men.rename(index={'Attention Deficit Disorder (ADHD)': 'ADHD',
                                  'Irritable Bowel Syndrome (IBS)': 'IBS'}, inplace=True)
    all_results_men.rename(columns={'Sleep Quality at baseline\n': 'Sleep test measures at baseline',
                                    'HRV at baseline\n': 'PRV at baseline'}, inplace=True)
    diffs_women['max'] = diffs_women.max(axis=1)
    all_results_women = pd.concat([all_results_women, diffs_women['max']], axis=1).sort_values(by='max',
                                                                                               ascending=False)
    all_results_women.rename(index={'Attention Deficit Disorder (ADHD)': 'ADHD',
                                    'G6PD': 'G6PD deficiency'}, inplace=True)
    all_results_women.rename(columns={'Sleep Quality at baseline\n': 'Sleep test measures at baseline',
                                      'HRV at baseline\n': 'PRV at baseline'}, inplace=True)

    # Figure
    title = f'AUCs for predicting medical diagnoses'
    fig = plt.figure(figsize=(10, 6))
    sns.set_style('white')
    # Males
    ax1 = fig.add_subplot(211)
    sns.heatmap(all_results_men.drop('max', axis=1).transpose(),
                cmap='Blues',
                vmin=0.5,
                vmax=0.6,
                linewidths=.75,
                xticklabels=True,
                # annot= True,
                cbar=True,
                cbar_kws={'label': 'Median AUC'},
                ax=ax1)
    ax1.set_title('Male')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, ha='right', fontsize=12)
    # Females
    ax2 = fig.add_subplot(212)
    sns.heatmap(all_results_women.drop('max', axis=1).transpose(),
                cmap='Reds',
                vmin=0.5,
                vmax=0.6,
                linewidths=.75,
                xticklabels=True,
                # annot= True,
                cbar=True,
                cbar_kws={'label': 'Median AUC'},
                ax=ax2)
    ax2.set_title('Female')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, ha='right', fontsize=12)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path, title + '.png'), dpi=DPI)
    plt.show()
    print(f'Figure saved in {path}')


if __name__ == '__main__':
    my_dir = MY_DIR
    special_folder = '/medical_diagnoses_associations'
    datasets_to_run = ['_women', '_men']  # choose from : '_women', '_men', '_all'
    models_to_run = ['Logit']
    body_systems = [
        'sleep_quality_avg',
        'hrv_avg',
    ]
    body_system_target = 'baseline_diagnoses_nastya'

    # Analyse results
    for from_dataset in body_systems:
        print('>>> Regression score distribution Vs baseline')
        body_system_feature = f'baseline_{from_dataset}'
        suffixes = [dataset.replace('_', '') for dataset in datasets_to_run]
        ticket_list = []
        model_type = models_to_run[0]
        model_score_files = {suffix: f'{model_type}_{suffix}scores_df.csv' for suffix in suffixes}
        target_group = None
        dataset_to_name = load_dataset_to_name()
        dir_path = os.path.join(my_dir + special_folder, f'from_{body_system_feature}',
                                'regressions_results',
                                f'{body_system_feature}_and_{body_system_target}')
        tmp_dict = {f'Age_Gender_BMI_and_{body_system_target}': dataset_to_name['Age_Gender_BMI'],
                    f'{body_system_feature}_and_{body_system_target}': dataset_to_name[body_system_feature]}
        for name, file in model_score_files.items():
            find_significant_predictions(name, dir_path, file, tmp_dict, target_group, f'from_{body_system_feature}')

    create_figure_for_paper(my_dir + special_folder, body_system_target)
    print('<<< Done')
