import matplotlib
from scipy import stats

from Fig1 import calc_mea_nutrient_trend
from Config import *
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


p_value = 0.1
Ns = ['TP', 'TN', 'TN2TP', 'Chla']
mea_data_dir = 'Data/MeaData'
region_colors = ['lightgrey', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']
save_dir = 'Results/Fig2'


def plt_climate_change_nutrient_reduction_patterns():
    # Climate change - Sigmoid shape
    x_array = np.linspace(-1.5, 1.5, 1000)
    y = 0.9 / (1 + np.exp(-4 * x_array))
    plt.figure(figsize=(2, 2))
    plt.plot(x_array, y, c='grey', lw=3, alpha=0.5)
    plt.xlim([-1.6, 1.7])
    plt.xticks([-1.5, -0.5, 0.5, 1.5], [0, 10, 20, 30])
    plt.ylim([-0.05, 1.1])
    plt.yticks([])
    ax = plt.gca()
    ax.fill_between(
        [-1.6, 1.7],
        [-0.05, -0.05],
        [1.1, 1.1],
        color='none',
        facecolor='#e2afaa',
        alpha=0.4,
        interpolate=True,
        linewidth=0.0
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Climate change pattern - Sigmoid.png'), dpi=300., transparent=True)
    plt.cla()


    # Nutrient reduction - Monod shape
    x_array = np.linspace(0, 1.5, 1000)
    y = 2 / (1 + np.exp(-4 * (x_array - 1.5)))
    plt.figure(figsize=(2, 2))
    plt.plot(x_array[::-1] + 0.2, y + 0.1, c='grey', lw=3, alpha=0.5)
    plt.xlim([0.1, 1.75])
    plt.xticks([0.15, 0.65, 1.15, 1.65], [0.15, 0.1, 0.05, 0.0][::-1])
    plt.ylim([0, 1.2])
    plt.yticks([])
    ax = plt.gca()
    ax.fill_between(
        [0.1, 1.75],
        [0, 0],
        [1.2, 1.2],
        color='none',
        facecolor='#5db3cb',
        alpha=0.4,
        interpolate=True,
        linewidth=0.0
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Nutrient reduction pattern - Monod.png'), dpi=300., transparent=True)
    plt.cla()


    # Climate change + Nutrient reduction
    plt.figure(figsize=(2, 2))
    ax = plt.gca()
    ax.fill_between(
        [0, 1],
        [0, 0],
        [0, 1],
        color='none',
        facecolor='#e2afaa',
        alpha=0.4,
        interpolate=True,
        linewidth=0.0
    )
    ax.fill_between(
        [0, 1],
        [0, 1],
        [1, 1],
        color='none',
        facecolor='#5db3cb',
        alpha=0.4,
        interpolate=True,
        linewidth=0.0
    )
    plt.xlim([0, 1.05])
    plt.xticks([])
    plt.ylim([0, 1.05])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Merge pattern.png'), dpi=300., transparent=True)
    plt.cla()


def plt_chla_trend_box_on_nutrient_and_temperature_threshold():
    trend_df, significant_df = calc_mea_nutrient_trend()
    trend_df *= 10.  # trend with unit (.../year) to (.../decade)
    lake_mean_attributes_df = pd.read_csv('Data/Lake mean attributes.csv', index_col='Index')
    lake_mean_attributes_df.loc[:, trend_df.columns] = trend_df.loc[lake_mean_attributes_df.index, :].values


    decline_loc, increase_loc = 1, 2
    lw = 0.25
    y_ticks = [-0.1, -0.05, 0, 0.05, 0.1]


    threshed_dic = {
        'T_Ave': 15,
        'TP_last_year': 0.02,
        'TP_trend': 0.0,
        'TN_trend': 0.0,
    }
    threshed_ls = [
        ['T_Ave'],

        ['TP_trend', 'T_Ave-lower'],
        ['TP_trend', 'T_Ave-upper'],

        ['TP_last_year', 'T_Ave-lower', 'TP_trend-lower'],
        ['TP_last_year', 'T_Ave-lower', 'TP_trend-upper'],
        ['TP_last_year', 'T_Ave-upper', 'TP_trend-lower'],
        ['TP_last_year', 'T_Ave-upper', 'TP_trend-upper'],
    ]


    fig, axes = plt.subplots(1, 7, sharex='col', figsize=(7, 3))
    for threshed_i, thresheds in enumerate(threshed_ls):
        plt.sca(axes[threshed_i])
        ax = plt.gca()

        threshed_name = thresheds[0]
        lower_idx = (lake_mean_attributes_df[threshed_name] < threshed_dic[threshed_name])
        upper_idx = (lake_mean_attributes_df[threshed_name] > threshed_dic[threshed_name])

        for threshed in thresheds[1:]:
            split_idx = threshed.index('-')
            threshed_name, threshed_part = threshed[: split_idx], threshed[split_idx + 1:]
            if threshed_part == 'upper':
                lower_idx &= (lake_mean_attributes_df[threshed_name] > threshed_dic[threshed_name])
                upper_idx &= (lake_mean_attributes_df[threshed_name] > threshed_dic[threshed_name])
            else:
                lower_idx &= (lake_mean_attributes_df[threshed_name] < threshed_dic[threshed_name])
                upper_idx &= (lake_mean_attributes_df[threshed_name] < threshed_dic[threshed_name])

        lower_chla_trend_values, upper_chla_trend_values = lake_mean_attributes_df[lower_idx]['Chla_trend'].values, lake_mean_attributes_df[upper_idx]['Chla_trend'].values

        plt.boxplot(
            [-lower_chla_trend_values, -upper_chla_trend_values],
            positions=[decline_loc, increase_loc],
            patch_artist=True, showfliers=False, showcaps=False,
            medianprops={'color': 'k', 'linewidth': lw * 4},
            flierprops={'markeredgewidth': lw * 2, 'markersize': 2},
            widths=0.6,
            boxprops={'linewidth': lw * 4, 'color': 'k', 'facecolor': 'none', 'alpha': 1},
            whiskerprops={'linewidth': 0, 'color': 'k'},
            capprops={'linewidth': 0, 'color': 'k'},
            zorder=0,
        )

        plt.scatter(
            np.random.normal(decline_loc, 0.05, len(lower_chla_trend_values)),
            -lower_chla_trend_values,
            marker='o',
            s=12,
            linewidths=lw,
            facecolors=region_colors[0],  # 'none'
            edgecolors='k',
            alpha=1,
            zorder=1,
        )
        plt.scatter(
            np.random.normal(increase_loc, 0.05, len(upper_chla_trend_values)),
            -upper_chla_trend_values,
            marker='o',
            s=12,
            linewidths=lw,
            facecolors=region_colors[0],  # 'none'
            edgecolors='k',
            alpha=1,
            zorder=1,
        )

        t, p = stats.ttest_1samp(lower_chla_trend_values, popmean=0)
        print('One sample T Test \t Variable:{} \t Low Mean:{:.4f} \t P:{:.4f}'.format(thresheds, lower_chla_trend_values.mean(), p))
        if p < p_value:
            plt.scatter(
                [decline_loc],
                [(y_ticks[0] + y_ticks[1]) * 0.5],
                marker='+' if lower_chla_trend_values.mean() > 0 else '|',
                s=48 if threshed_i < 3 else 64,
                linewidths=2,
                facecolors='k',
                edgecolors='k',
                alpha=1,
            )
        t, p = stats.ttest_1samp(upper_chla_trend_values, popmean=0)
        print('One sample T Test \t Variable:{} \t Upp Mean:{:.4f} \t P:{:.4f}'.format(thresheds, upper_chla_trend_values.mean(), p))
        if p < p_value:
            plt.scatter(
                [increase_loc],
                [(y_ticks[0] + y_ticks[1]) * 0.5],
                marker='+' if upper_chla_trend_values.mean() > 0 else '|',
                s=48 if threshed_i < 3 else 64,
                linewidths=2,
                facecolors='k',
                edgecolors='k',
                alpha=1,
            )

        t, p = stats.ttest_ind(lower_chla_trend_values, upper_chla_trend_values)
        if p < p_value:
            s = 'T Test Thresheds:{} N_cls0:{} N_cls1:{} P:{:.4f}' \
                .format(thresheds, len(lower_chla_trend_values), len(upper_chla_trend_values), p)
            # print(s)
            plt.plot(
                [decline_loc, increase_loc],
                [y_ticks[-2], y_ticks[-2]],
                c='k',
                lw=lw * 4,
            )
            plt.text((decline_loc + increase_loc) * 0.5, y_ticks[-2] - 0.005, '*', fontsize=20 if threshed_i < 3 else 28, ha="center")

        plt.text(decline_loc, y_ticks[-1] - 0.02, '{}'.format(len(lower_chla_trend_values)), fontsize=8 if threshed_i < 3 else 10, ha="center", rotation=-90)
        plt.text(increase_loc, y_ticks[-1] - 0.02, '{}'.format(len(upper_chla_trend_values)), fontsize=8 if threshed_i < 3 else 10, ha="center", rotation=-90)

        plt.xticks([], [])
        plt.xticks([decline_loc, increase_loc], [])
        plt.xlim([0.5, 2.5])
        plt.yticks(y_ticks, y_ticks[::-1])
        plt.ylim([y_ticks[0], y_ticks[-1]])
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=lw, alpha=0.4)
        ax.tick_params(axis='both', which='major', width=lw * 0.5, length=3, labelsize=8)
        ax.tick_params(axis='y', which='major', labelrotation=-90)
        for spine in ax.spines.values():
            spine.set_linewidth(lw * 0.5)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.1)
    plt.savefig(os.path.join(save_dir, 'Chla trend box at temperature and nutrient threshold.png'), dpi=300., transparent=False)
    plt.cla()


if __name__ == '__main__':
    # plt_climate_change_nutrient_reduction_patterns()
    plt_chla_trend_box_on_nutrient_and_temperature_threshold()
