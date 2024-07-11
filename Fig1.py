import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk
from osgeo import ogr

from Config import *
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


p_value = 0.1
Ns = ['TP', 'TN', 'TN2TP', 'Chla']
mea_data_dir = 'Data/MeaData'
region_colors = ['lightgrey', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']
save_dir = 'Results/Fig1'
lake_point_shapefile_path = 'Data/GeoData/Shapefiles/Lake_point.shp'


def calc_mea_nutrient_trend():
    case_ids = list(UsedCases_dic.keys())
    trend_df = pd.DataFrame(index=case_ids)
    significant_df = pd.DataFrame(index=case_ids)


    for case_id in case_ids:
        N_dic = {}
        for N in Ns:
            if N != 'TN2TP':
                N_data_path = os.path.join(mea_data_dir, '{}-{}-Mean.csv'.format(case_id, N))
                N_data_df = pd.read_csv(N_data_path, index_col=0, header=0)
                N_data_df.index = pd.to_datetime(N_data_df.index)
                N_year_mean_ss = N_data_df.groupby(N_data_df.index.year)[N].mean().interpolate(method='linear', limit_direction='both')
                N_year_mean_ss = N_year_mean_ss.dropna()
                N_dic[N] = N_year_mean_ss
            else:
                N_year_mean_ss = N_dic['TN'] / N_dic['TP']

            sen_slope = mk.sens_slope(N_year_mean_ss)[0]
            significant_test_results = mk.original_test(N_year_mean_ss, alpha=p_value).trend    # ['no trend', 'increasing', 'decreasing']

            trend_df.loc[case_id, '{}_trend'.format(N)] = sen_slope
            significant_df.loc[case_id, '{}_trend'.format(N)] = 0 if significant_test_results == 'no trend' else 1

    trend_df.to_csv(os.path.join(save_dir, 'Trend.csv'), index_label='Index')
    significant_df.to_csv(os.path.join(save_dir, 'Trend significant.csv'), index_label='Index')
    return trend_df, significant_df


def plt_regional_trend_box():
    y_tick_dic = {
        "Chla": [-0.04, -0.02, 0, 0.02, 0.04],
        "TP": [-0.1, -0.05, 0, 0.05, 0.1],
        "TN": [-4, -2, 0, 2, 4],
        "TN2TP": [-100, -50, 0, 50, 100]
    }
    lw = 0.25


    trend_df, significant_df = calc_mea_nutrient_trend()
    trend_df *= 10.     # trend with unit (.../year) to (.../decade)


    for N in Ns:
        plt.figure(figsize=(2.2, 1.5))
        y_ticks = y_tick_dic[N]
        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            values = trend_df.loc[region_lake_ids, '{}_trend'.format(N)].values

            plt.boxplot(
                values,
                positions=[region_id],
                patch_artist=True, showfliers=False, showcaps=False,
                medianprops={'color': 'k', 'linewidth': lw * 4},
                flierprops={'markeredgewidth': lw, 'markersize': 3},
                widths=0.6,
                boxprops={'linewidth': lw * 4, 'color': 'k', 'facecolor': 'none', 'alpha': 1},
                whiskerprops={'linewidth': 0, 'color': 'k'},
                capprops={'linewidth': lw * 3, 'color': 'k'},
                zorder=0,
            )

            plt.scatter(
                np.random.normal(region_id, 0.06, len(values)),
                values,
                marker='o',
                s=10,
                linewidths=0.25,
                facecolors=region_colors[region_id],  # 'none'
                edgecolors='k',
                alpha=1,
                zorder=1,
            )

            t, p = stats.ttest_1samp(values, popmean=0)
            print('One sample T Test \t Variable:{} \t RegionID:{} \t Mean:{:.4f} \t Median:{:.4f} \t P:{:.4f}'.format(N, region_id, values.mean(), np.median(values), p))
            if p <= p_value:
                plt.scatter(
                    [region_id],
                    [(y_ticks[0] + y_ticks[1]) * 0.5],
                    marker='+' if values.mean() > 0 else '_',
                    s=64,
                    linewidths=2,
                    facecolors='k',
                    edgecolors='k',
                    alpha=1,
                    zorder=2,
                )


        region_names = list(RegionName_dic.values())
        plt.xticks(range(0, len(region_names) + 1), ['All'] + region_names)
        plt.yticks(y_ticks)
        plt.ylim([y_ticks[0], y_ticks[-1]])
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        plt.tight_layout()
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=lw, alpha=0.4)
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.98, top=0.97)
        plt.savefig(os.path.join(save_dir, 'Map sub box fig - {}.png'.format(N)), dpi=300., transparent=False)
        plt.cla()


def plt_regional_nutrient_trend_vs_chla_trend_box():
    decline_loc, increase_loc = 1, 2
    region_box_bias = [-0.3, -0.15, 0, 0.15, 0.3]
    lw = 0.25
    box_width = 0.1
    y_ticks = [-0.1, -0.05, 0, 0.05, 0.1]

    trend_df, significant_df = calc_mea_nutrient_trend()
    trend_df *= 10.  # trend with unit (.../year) to (.../decade)


    fig, axes = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(7.5, 2.5))
    for i_N, N in enumerate(Ns[:-1]):
        plt.sca(axes[i_N])
        ax = plt.gca()

        row_line_y = y_ticks[1]
        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            region_trend_df = trend_df.loc[region_lake_ids]
            chla_trends_in_nutrient_decreasing = region_trend_df[region_trend_df['{}_trend'.format(N)] < 0]['Chla_trend'].values
            chla_trends_in_nutrient_increasing = region_trend_df[region_trend_df['{}_trend'.format(N)] > 0]['Chla_trend'].values

            plt.boxplot(
                [chla_trends_in_nutrient_decreasing, chla_trends_in_nutrient_increasing],
                positions=[decline_loc + region_box_bias[region_id], increase_loc + region_box_bias[region_id]],
                patch_artist=True, showfliers=False, showcaps=False,
                medianprops={'color': 'k', 'linewidth': lw * 4},
                flierprops={'markeredgewidth': lw * 2, 'markersize': 2},
                widths=box_width,
                boxprops={'linewidth': lw * 4, 'color': 'k', 'facecolor': 'none', 'alpha': 1},
                whiskerprops={'linewidth': 0, 'color': 'k'},
                capprops={'linewidth': lw * 2, 'color': 'k'},
                zorder=0,
            )

            plt.scatter(
                np.random.normal(decline_loc + region_box_bias[region_id], 0.005, len(chla_trends_in_nutrient_decreasing)),
                chla_trends_in_nutrient_decreasing,
                marker='o',
                s=10,
                linewidths=lw,
                facecolors=region_colors[region_id],  # 'none'
                edgecolors='k',
                alpha=1,
                zorder=1,
            )
            plt.scatter(
                np.random.normal(increase_loc + region_box_bias[region_id], 0.005, len(chla_trends_in_nutrient_increasing)),
                chla_trends_in_nutrient_increasing,
                marker='o',
                s=10,
                linewidths=lw,
                facecolors=region_colors[region_id],  # 'none'
                edgecolors='k',
                alpha=1,
                zorder=1,
            )

            t, p = stats.ttest_1samp(chla_trends_in_nutrient_decreasing, popmean=0)
            print('One sample T Test \t Variable:{} \t RegionID:{} \t De Mean:{:.4f} \t P:{:.4f}'.format(N, region_id, chla_trends_in_nutrient_decreasing.mean(), p))
            if p < p_value:
                plt.scatter(
                    [decline_loc + region_box_bias[region_id]],
                    [(y_ticks[-2] + y_ticks[-1]) * 0.5],
                    marker='+' if chla_trends_in_nutrient_decreasing.mean() > 0 else '_',
                    s=48,
                    linewidths=2,
                    facecolors='k',
                    edgecolors='k',
                    alpha=1,
                )

            t, p = stats.ttest_1samp(chla_trends_in_nutrient_increasing, popmean=0)
            print('One sample T Test \t Variable:{} \t RegionID:{} \t In Mean:{:.4f} \t P:{:.4f}'.format(N, region_id, chla_trends_in_nutrient_increasing.mean(), p))
            if p < p_value:
                plt.scatter(
                    [increase_loc + region_box_bias[region_id]],
                    [(y_ticks[-2] + y_ticks[-1]) * 0.5],
                    marker='+' if chla_trends_in_nutrient_increasing.mean() > 0 else '_',
                    s=48,
                    linewidths=2,
                    facecolors='k',
                    edgecolors='k',
                    alpha=1,
                )

            t, p = stats.ttest_ind(chla_trends_in_nutrient_decreasing, chla_trends_in_nutrient_increasing)
            s = 'T Test Variable:{} RegionID:{} De:{} In:{} P:{:.4f}' \
                .format(N, region_id, len(chla_trends_in_nutrient_decreasing), len(chla_trends_in_nutrient_increasing), p)
            print(s)
            if p < p_value:
                plt.plot(
                    [decline_loc + region_box_bias[region_id], increase_loc + region_box_bias[region_id]],
                    [row_line_y, row_line_y],
                    c='k',
                    lw=lw * 4,
                )
                plt.text((decline_loc + region_box_bias[region_id] + increase_loc + region_box_bias[region_id]) * 0.5, row_line_y - 0.02, '*', fontsize=20, ha="center")
                row_line_y -= 0.015

            plt.text(decline_loc + region_box_bias[region_id] - 0.01, y_ticks[0] + 0.01, '{}'.format(len(chla_trends_in_nutrient_decreasing)), fontsize=8, ha="center")
            plt.text(increase_loc + region_box_bias[region_id] - 0.01, y_ticks[0] + 0.01, '{}'.format(len(chla_trends_in_nutrient_increasing)), fontsize=8, ha="center")

        plt.xticks([], [])
        plt.xticks([decline_loc, increase_loc], ['{} trend < 0'.format(N if N != 'TN2TP' else 'TN:TP'), 'Others'])
        plt.plot([1.5, 1.5], [y_ticks[0], y_ticks[-1]], lw=lw * 0.5, c='k', ls='--')
        plt.xlim([0.5, 2.5])
        plt.yticks(y_ticks)
        plt.ylim([y_ticks[0], y_ticks[-1]])
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.25, alpha=0.4)
        ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        if i_N == 0:
            plt.ylabel('Chl$\it{a}$ trend (mg L$^{-1}$ decade$^{-1}$)', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig(os.path.join(save_dir, 'Chla trend box in different nutrient trends.png'), dpi=300., transparent=False)
    plt.cla()


def load_nutrient_trend_to_map():
    trend_df, significant_df = calc_mea_nutrient_trend()
    trend_df *= 10.  # trend with unit (.../year) to (.../decade)


    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(lake_point_shapefile_path, 1)


    layer = dataset.GetLayer()
    layer_defn = layer.GetLayerDefn()
    for N in Ns:
        trd_field_name, sig_field_name = 'trd_{}'.format(N), 'sig_{}'.format(N)
        if layer_defn.GetFieldIndex(sig_field_name) < 0:
            new_int_field = ogr.FieldDefn(sig_field_name, ogr.OFTInteger)
            layer.CreateField(new_int_field)
        if layer_defn.GetFieldIndex(trd_field_name) < 0:
            new_float_field = ogr.FieldDefn(trd_field_name, ogr.OFTReal)
            layer.CreateField(new_float_field)


    feature = layer.GetNextFeature()
    while feature:
        lake_id = str(feature.GetField('LakeID'))
        for N in Ns:
            feature.SetField('trd_{}'.format(N), trend_df.loc[lake_id, '{}_trend'.format(N)])
            feature.SetField('sig_{}'.format(N), int(significant_df.loc[lake_id, '{}_trend'.format(N)]))
            layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    layer = None
    dataset = None


if __name__ == '__main__':
    plt_regional_trend_box()
    plt_regional_nutrient_trend_vs_chla_trend_box()
    # load_nutrient_trend_to_map()
