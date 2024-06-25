# Setup
import warnings
import pandas as pd

from utils import *
from scipy.stats import sem, t


def mean_ci_95(data):
    n = len(data)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + 0.95) / 2., n - 1)
    return m, m - h, m + h


if __name__ == '__main__':

    my_dir = MY_DIR
    os.chdir(mkdirifnotexists(os.path.join(my_dir, 'Logs')))
    special_folder = '/clinical_OSA_associations'
    datasets_to_run = ['_men', '_women']  # choose from : '_women', '_men', '_all'
    models_to_run = ['Logit']
    body_systems = [
            'Age_Gender_BMI_VAT',
            'body_composition',
            'bone_density',
            'cardiovascular',
            'frailty',
            'glycemic_status',
            'hematopoietic',
            'immune_system',
            'liver',
            'renal_function',
            'MBfamily2',
            'MBpathways',
            'mental',
            'diet',
            'medications',
            'blood_lipids'
    ]
    target = 'clinical_OSA'

    # Check significance vs predictions from Age, BMI and VAT for each body system:
    if 0:
        print('>>> Classification score distribution Vs baseline')
        suffixes = [dataset.replace('_', '') for dataset in datasets_to_run]
        for from_dataset in body_systems:
            model_score_files = {suffix: f'{models_to_run[0]}_{suffix}scores_df.csv' for suffix in suffixes}
            target_group = None
            dataset_to_name = load_dataset_to_name()
            dir_path = os.path.join(my_dir + special_folder, f'from_{from_dataset}',
                                    'regressions_results',
                                    f'{from_dataset}_and_{target}')
            tmp_dict = {f'Age_Gender_BMI_VAT_and_{target}': dataset_to_name['Age_Gender_BMI_VAT'],
                        f'{from_dataset}_and_{target}': dataset_to_name[from_dataset]}
            for name, file in model_score_files.items():
                find_significant_predictions(name, dir_path, file, tmp_dict, target_group, f'from_{from_dataset}')

    # Load significance results and create figures:
    sexes = [dataset.replace('_', '') for dataset in datasets_to_run]
    sig_dict = {sex: pd.Series(index=body_systems) for sex in sexes}
    body_systems_copy = body_systems.copy()
    body_systems_copy.remove('Age_Gender_BMI_VAT')
    for sex in sexes:
        for body_system in body_systems_copy:
            file = os.path.join(my_dir+special_folder, f'from_{body_system}', 'regressions_results', f'{body_system}_and_{target}',
                                f'plots_{models_to_run[0]}', f'median_scores_filtered-{sex}_dataset.csv')
            df = pd.read_csv(file, index_col=0)
            if df.shape[0] == 0:
                sig_dict[sex][body_system] = 0
            else:
                sig_dict[sex][body_system] = 1
        sig_dict[sex]['Age_Gender_BMI_VAT'] = 0
        sig_dict[sex]['glycemic_status'] = 0

    # Combine the data into a single DataFrame for easier plotting
    auc_df_females = pd.DataFrame(columns=body_systems)
    auc_df_males = pd.DataFrame(columns=body_systems)
    for body_system in body_systems:
        for i, auc_df in enumerate([auc_df_males, auc_df_females]):
            tmp = pd.read_csv(os.path.join(my_dir + special_folder, f'from_{body_system}', 'regressions_results',
                                           f'{body_system}_and_{target}',
                                           f'{body_system}_and_{target}_Logit{datasets_to_run[i]}scores_df.csv'),
                              index_col=[0])
            auc_df[body_system] = tmp[target]
    df_females_melted = auc_df_females.melt(var_name='body system', value_name='AUC')
    df_females_melted['sex'] = 'Females'
    df_males_melted = auc_df_males.melt(var_name='body system', value_name='AUC')
    df_males_melted['sex'] = 'Males'
    combined_df = pd.concat([df_males_melted, df_females_melted])
    ordered_labels = combined_df.groupby('body system')['AUC'].mean().sort_values().index
    ordered_labels = ['Age_Gender_BMI_VAT'] + [label for label in ordered_labels if label != 'Age_Gender_BMI_VAT']
    dict_for_ticks = load_dataset_to_name()
    dict_for_ticks['MBpathways'] = f'Gut MB metabolic\npathways'
    dict_for_ticks['cardiovascular'] = 'Cardiovascular\nsystem'
    ticksize = 15
    colors = {'Males': sns.color_palette('Paired')[1], 'Females': sns.color_palette('rocket')[4]}

    # Create the forest plot
    stats = combined_df.groupby(['body system', 'sex'])['AUC'].apply(mean_ci_95).unstack()
    stats = stats.loc[ordered_labels]
    stats.to_csv(os.path.join(MY_DIR + special_folder, 'AUC_stats.csv'))
    fig, ax = plt.subplots(figsize=(7, 10))
    systems = stats.index.get_level_values('body system').unique()
    y_positions = np.arange(len(systems))
    dict_for_sig = {'Females': 'women', 'Males': 'men'}
    for sex, color, offset in zip(['Females', 'Males'],
                                  [sns.color_palette('rocket')[4], sns.color_palette('Paired')[1]], [-0.1, 0.1]):
        means = stats[sex].apply(lambda x: x[0])
        cis_2_5 = stats[sex].apply(lambda x: x[1])
        cis_97_5 = stats[sex].apply(lambda x: x[2])
        ax.errorbar(means, y_positions + offset, xerr=[(means - cis_2_5), (cis_97_5 - means)],
                    fmt='o', label=sex, color=color, capsize=5)
        sig_dict[dict_for_sig[sex]] = sig_dict[dict_for_sig[sex]][ordered_labels]
        for j, (x, y, sig) in enumerate(zip(cis_97_5, y_positions, sig_dict[dict_for_sig[sex]])):
            if sig:
                ax.text(x + 0.002, y + offset - 0.1, '*', color='black', fontsize=15, ha='right', va='center')
    plt.axvline(0.5, color='k', linestyle='--')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels=[dict_for_ticks[body_system] for body_system in systems], fontsize=ticksize)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=ticksize)
    ax.set_xlabel('AUC (mean and 95% confidence interval)', fontsize=ticksize)
    ax.legend(fontsize=ticksize)
    plt.tight_layout()
    plt.savefig(os.path.join(MY_DIR + special_folder, 'Forest Plot for clinical OSA predictions.png'), dpi=300)
    plt.show()