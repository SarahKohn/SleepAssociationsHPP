import random
import matplotlib as mpl

from utils import *

# Define directories path:
RESULTS_PATH = os.path.join(MY_DIR, 'correlations_results')
random.seed(10)


def plot_correlations_donut(body_systems: list):
    print('> Create a nested pie chart for correlations')
    title_fontsize = FONTSIZE + 5
    label_fontsize = FONTSIZE
    scale = (-0.6, 0.6)

    ## Create a concatenated dataframe
    dfs = pd.DataFrame()
    top_corr_values = pd.Series(dtype='float64')
    for body_system in body_systems:
        filename = os.path.join(RESULTS_PATH, f'pheno_sleep_avg_and_{body_system}_corr_df.csv')
        mask_file = os.path.join(RESULTS_PATH, f'pheno_sleep_avg_and_{body_system}_mask_df.csv')
        df = pd.read_csv(filename, index_col=0)
        mask = pd.read_csv(mask_file, index_col=0)
        features = df.columns.to_list()
        if ('gender' in features) and (body_system != 'Age_Gender_BMI'):
            df = df.drop(columns=['gender'])
            mask = mask.drop(columns=['gender'])
        if ('bmi' in features) and (body_system != 'Age_Gender_BMI'):
            df = df.drop(columns=['bmi'])
            mask = mask.drop(columns=['bmi'])
        if body_system == 'lifestyle':
            df = df.drop(columns=['snoring_no'])
            mask = mask.drop(columns=['snoring_no'])
        df.replace(0, np.nan, inplace=True)
        df[mask] = np.nan
        df.dropna(how='all', axis=1, inplace=True)
        targets = df.index.to_list()
        if 'AHI' in targets:
            top_features = df.abs().transpose().sort_values(by='AHI', ascending=False)[0:10].index.to_list()
            df = df[top_features].transpose()
            top_corr_values[body_system] = df['AHI'].iloc[0]
        else:
            top_features = df.abs().transpose().sort_values(by='desaturations_mean_nadir', ascending=False)[0:10].index.to_list()
            df = df[top_features].transpose()
            top_corr_values[body_system] = df['desaturations_mean_nadir'][0]
        df['body_system'] = body_system
        if body_system == 'diet':
            df.index = df.index.str.cat([' intake'] * len(df.index), sep='')
        elif body_system == 'blood_lipids':
            df.index = df.index.str.slice(0, 15)
        dfs = pd.concat([dfs, df])
    category_order = top_corr_values.abs().sort_values(ascending=False).index.to_list()[::-1]
    dfs['body_system'] = pd.Categorical(dfs['body_system'], categories=category_order, ordered=True)
    dfs = dfs.sort_values(by=['body_system', 'AHI', 'desaturations_mean_nadir'], ascending=False)
    dfs.to_csv(os.path.join(RESULTS_PATH, 'results.csv'))

    ## Create the nested pie chart:
    df = pd.DataFrame()
    groups = dfs['body_system'].unique().tolist()
    for bs in groups:
        df = pd.concat((df, dfs[dfs['body_system'] == bs], pd.DataFrame(np.NaN, index=[''], columns=dfs.columns)), axis=0,
                       sort=False)
        df.iloc[-1, -1] = bs
    blank_len = int(dfs.shape[0] * 0.15)
    df = pd.concat((df, pd.DataFrame(np.NaN, index=[''] * blank_len, columns=dfs.columns)), axis=0, sort=False)
    group_sizes = []
    for i in groups:
        group_sizes += [df['body_system'].value_counts().loc[i] - 1, 1]
    group_sizes.pop()
    group_sizes.extend([blank_len])
    df = df.drop(columns=['body_system',
                          'saturation_mean',
                          'percent_of_supine_sleep',
                          'variability_between_sleep_stage_percents',
                          'hrv_time_rmssd_during_wake'])
    colors = get_qualitative_colors_without_reds_and_blues()
    bs_colors = []
    for color in colors:
        bs_colors += [color, (1, 1, 1, 1)]
    corr_cmap = plt.cm.RdBu_r  #coolwarm
    # correlations layers:
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot()
    radius = 2.0
    radius_step = 0.15
    mapping_features_dict = features_renaming_dict()
    mapping_features_dict.update(mbfamily_to_name())
    for layer in df.columns:
        if layer == 'AHI':
            labels = df.rename(index=mapping_features_dict).index
            labels = labels.str.replace('_', ' ')
            labels = labels.str.replace('bt', 'BT')
            labels = labels.str.split(pat=': ', n=1).str[-1]  # for MB pathways
            labels = labels.str.split(pat='(', n=1).str[0]  # shrink where parentheses
        else:
            labels = ['' for i in df.index]
        mypie2, texts = ax.pie([1 for i in range(df.shape[0])], radius=radius, labels=labels, rotatelabels=True,
                               labeldistance=1., textprops={'fontsize': label_fontsize},
                               colors=get_scale_colors([corr_cmap], df[layer], boundries=scale),
                               startangle=90)
        plt.setp(mypie2, width=radius_step, edgecolor='white')
        radius -= radius_step
    # body systems layer
    names_dict = load_dataset_to_name()
    names_dict['Age_Gender_BMI'] = 'Sex, Age & BMI'
    labels = [names_dict[bs] for bs in groups]
    mypie, _ = ax.pie(group_sizes, radius=radius, colors=bs_colors,
                      textprops=dict(color='black', fontsize=label_fontsize), startangle=90)
    plt.setp(mypie, width=radius_step / 2, edgecolor='white')
    if 1:  # put labels aside the pie portion in bboxes:
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.0)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center",
                  size=label_fontsize
                  )
        x_factor = [-0.2, 0.05, 0.35, 0.2, 0.8, 0.35, 0.55, 0.25, 0.2, 0.05, -0.05, 0.2, 0.4, 0.6, 0.5, 0.2, 0.7]
        x_factor = [val - 0.1 for val in x_factor]
        y_factor = [0.95, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95, 0.95, 0.85, 0.95, 0.85, 0.80, 0.80, 0.95, 0.95, 0.95,
                    0.95]
        for i, p in enumerate(mypie):
            if (i % 2) and (i != 0):
                continue
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {1: "left", -1: "right"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[int(i / 2)], xy=(x, y),
                        xytext=(x_factor[int(i / 2)] * np.sign(x), y_factor[int(i / 2)] * y),
                        horizontalalignment=horizontalalignment,
                        **kw)
    else:  # list the body systems in a legend:
        legend = ax.legend(mypie, labels,
                           title="Body Systems",
                           loc="center",
                           bbox_to_anchor=(0., 0, 1, 1.),
                           fontsize=label_fontsize, frameon=False)
        legend.get_title().set_fontsize(str(title_fontsize))
    # Save figure
    plt.savefig(os.path.join(RESULTS_PATH, 'nested_pie-correlations.png'), dpi=300, bbox_inches='tight')
    plt.show()

    ## Create circles legend:
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(8, 5))
    legend_names = {
        'snore_db_mean': 'Snoring level',
        'desaturations_mean_nadir': 'SpO2 nadir',
        'total_sleep_time': 'Sleep time',
        'sleep_efficiency': 'Sleep efficiency',
        'hrv_time_rmssd_during_night': 'HRV (during sleep time)'
    }
    df = df.rename(columns=legend_names)
    for i, legend in enumerate(df.columns):
        ax = axes[i]
        ax.text(0, 0.1, legend, fontsize=title_fontsize)
        ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'nested_pie_legend-correlations.png'), dpi=300)
    plt.close()

    ## Create color bar:
    fig, ax = plt.subplots(1, 1, figsize=(1, 5), layout='constrained')
    cb = mpl.colorbar.ColorbarBase(ax, cmap=corr_cmap,
                                   norm=mcolors.CenteredNorm(),
                                   ticks=[-0.99, 0, 1],
                                   orientation='vertical')
    cb.ax.tick_params(labelsize=title_fontsize)
    cb.ax.set_yticklabels([scale[0], 0, scale[1]])
    cb.ax.set_ylim([-0.99, 1.0])
    plt.savefig(os.path.join(RESULTS_PATH, 'nested_pie_cbar-correlations.png'), dpi=300)
    plt.close()
    print(f'Figure saved in {RESULTS_PATH}')
    pass


def get_qualitative_colors_without_reds_and_blues() -> list:
    original_colors = plt.get_cmap('tab20c')
    keep_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # remove blues and reds
    new_colors_1 = [original_colors(i) for i in keep_indices]
    original_colors = plt.get_cmap('Accent')
    keep_indices = [3, 5, 6]  # remove blues and reds
    new_colors_2 = [original_colors(i) for i in keep_indices]
    original_colors = plt.get_cmap('Set2')
    keep_indices = [3, 5]  # remove blues and reds
    new_colors_3 = [original_colors(i) for i in keep_indices]
    colors = new_colors_1 + new_colors_2 + new_colors_3
    random.shuffle(colors)
    return colors


if __name__ == '__main__':

    print('\n>>> Start')
    body_systems = [
        'Age_Gender_BMI',
        'body_composition',
        'bone_density',
        'cardiovascular',
        'frailty',
        'glycemic_status',
        'hematopoietic',
        'immune_system',
        'lifestyle',
        'liver',
        'renal_function',
        'MBfamily2',
        'MBpathways',
        'mental',
        'diet',
        'medications',
        'blood_lipids'
    ]
    plot_correlations_donut(body_systems)
    print('<<< Done')
