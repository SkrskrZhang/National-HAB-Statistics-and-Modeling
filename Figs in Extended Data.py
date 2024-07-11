import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import pymannkendall as mk
from osgeo import ogr
from copy import deepcopy
from pygam import LinearGAM, s, f
from scipy.stats import gaussian_kde
from scipy.stats import linregress
import matplotlib.dates as mdates
import seaborn as sns

from Config import *
from Fig1 import calc_mea_nutrient_trend
from NationalHABModel import Model
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


p_value = 0.1
Ns = ['TP', 'TN', 'TN2TP', 'Chla']
mea_data_dir = 'Data/MeaData'
region_colors = ['lightgrey', '#50730a', '#b70046', '#006c6e', '#bd9700']
save_dir = 'Results/Figs in Extended Data'
lake_point_shapefile_path = 'Data/GeoData/Shapefiles/Lake_point.shp'


def fig1to4_nutrient_trend_box():
    case_ids = list(UsedCases_dic.keys())
    n_row, n_col = 25, 4
    vars_dic = {
        'Chla': [0.02, 'mg L$^{-1}$ decade$^{-1}$'],
        'TP': [0.05, 'mg L$^{-1}$ decade$^{-1}$'],
        'TN': [0.5, 'mg L$^{-1}$ decade$^{-1}$'],
        'TN2TP': [50, 'decade$^{-1}$'],
    }
    lw = 0.25
    inc_c, dec_c = 'r', 'b'
    sig_ls, nosig_ls = '-', '--'


    trend_df, significant_df = calc_mea_nutrient_trend()
    trend_df *= 10.  # trend with unit (.../year) to (.../decade)

    for i_N, N in enumerate(Ns):
        fig, axes = plt.subplots(n_row, n_col, figsize=(10, 12), sharex='col')
        axes = axes.flatten()
        i = -1
        for region_id, region_name in RegionName_dic.items():
            region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]

            for case_id in region_case_ids:
                i += 1
                plt.sca(axes[i])
                ax = plt.gca()

                if N != 'TN2TP':
                    N_data_path = os.path.join(mea_data_dir, '{}-{}-Mean.csv'.format(case_id, N))
                    N_data_df = pd.read_csv(N_data_path, index_col=0, header=0)
                    N_data_df.index = pd.to_datetime(N_data_df.index)
                else:
                    TN_data_path = os.path.join(mea_data_dir, '{}-{}-Mean.csv'.format(case_id, 'TN'))
                    TN_data_df = pd.read_csv(TN_data_path, index_col=0, header=0)
                    TN_data_df.index = pd.to_datetime(TN_data_df.index)
                    TP_data_path = os.path.join(mea_data_dir, '{}-{}-Mean.csv'.format(case_id, 'TP'))
                    TP_data_df = pd.read_csv(TP_data_path, index_col=0, header=0)
                    TP_data_df.index = pd.to_datetime(TP_data_df.index)
                    N_data_df = pd.DataFrame(index=TP_data_df.index)
                    N_data_df.loc[:, N] = TN_data_df['TN'] / TP_data_df['TP']
                    N_data_df = N_data_df[N_data_df[N] < 1000]

                N_data_df = N_data_df.dropna(axis=0, how='all')
                N_year_mean_ss = N_data_df.groupby(N_data_df.index.year)[N].mean().interpolate(method='linear', limit_direction='both')
                N_year_mean_ss = N_year_mean_ss.dropna()
                sen_slope = mk.sens_slope(N_year_mean_ss)[0]
                significant_test_results = mk.original_test(N_year_mean_ss, alpha=p_value).trend    # ['no trend', 'increasing', 'decreasing']

                for year in N_data_df.index.year.unique():
                    year_df = N_data_df[N_data_df.index.year == year]
                    year_values = year_df.iloc[:, 0].values
                    plt.boxplot(
                        year_values,
                        positions=[year],
                        patch_artist=True, showfliers=True, manage_ticks=False, showmeans=True, meanline=True,
                        medianprops={'color': 'k', 'linewidth': 0},
                        meanprops={'color': region_colors[region_id + 1], 'linewidth': lw * 3, 'linestyle': '-'},
                        flierprops={'markeredgewidth': lw, 'markersize': 2, 'alpha': 0.5, 'color': region_colors[region_id + 1]},
                        widths=0.5,
                        boxprops={'linewidth': lw * 2, 'color': region_colors[region_id + 1], 'facecolor': 'none'},
                        whiskerprops={'linewidth': lw * 2, 'color': region_colors[region_id + 1]},
                        capprops={'linewidth': lw * 2, 'color': region_colors[region_id + 1]},
                        zorder=0,
                    )

                intercept = N_year_mean_ss.values.mean() - (sen_slope * np.array(N_year_mean_ss.index).mean())
                plt.plot(
                    N_year_mean_ss.index, np.array(N_year_mean_ss.index) * sen_slope + intercept,
                    c='r' if sen_slope > 0 else 'b',
                    lw=lw * 5,
                    ls='--' if significant_test_results == 'no trend' else '-',
                    zorder=1,
                )
                plt.text(0.025, 0.8, AllCasesEngName_dic[case_id], transform=ax.transAxes, fontsize=6)
                plt.text(0.025, 0.5, 'Trend: {:.4f} {}'.format(sen_slope * 10.0, vars_dic[N][-1]), transform=ax.transAxes, fontsize=6)
                plt.xlim([2004, 2023])
                plt.xticks([2005, 2010, 2015, 2020])
                plt.xlabel('')

                y_tick_idx = vars_dic[N][0]
                y_max = (N_data_df.values.max() // y_tick_idx) * y_tick_idx + y_tick_idx
                plt.ylim([0, y_max])
                if i < n_col:
                    plt.yticks([0, y_max / 2, y_max])
                else:
                    plt.yticks([0, y_max / 2])

                for spine in ax.spines.values():
                    spine.set_linewidth(lw)
                ax.tick_params(axis='both', which='major', width=0.25, length=4, labelsize=6)
                ax.tick_params(axis='both', which='minor', width=0.25, length=2)
                plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=0.25, alpha=0.4)

        for ax in axes[-3:]:
            ax.remove()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0.25)
        plt.savefig(os.path.join(save_dir, 'Fig{} nutrient trend box - {}.png'.format(i_N + 1, N)), dpi=300.)


def fig5_zonal_climate_change_box():
    relation_df = pd.read_csv(os.path.join('Data', 'Lake mean attributes.csv'), index_col=0, header=0)
    relation_df['TN2TP'] = relation_df['TN'] / relation_df['TP']
    trend_df, significant_df = calc_mea_nutrient_trend()
    relation_df.loc[trend_df.index, trend_df.columns] = trend_df.values
    relation_df[['Chla', 'TP', 'TN']] *= 1000  # scale mg to ug
    relation_df[['Chla', 'TP', 'TN', 'TN2TP']] = np.log10(relation_df[['Chla', 'TP', 'TN', 'TN2TP']])
    relation_df['SR_Sum'] *= 365.25  # scale daily average to annual average
    for c in relation_df.columns:
        if str(c).endswith('trend'):
            relation_df[c] *= 10  # scale trend in year-1 to decade-1
    region_colors = ['#d3d3d3', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']


    variables = ['T_Ave', 'SR_Sum', 'WS_Ave']
    variable_mean_y_ticks_dic = {
        'T_Ave': [0, 7.5, 15, 22.5, 30],
        'SR_Sum': [4000, 4700, 5400, 6100, 6800],
        'WS_Ave': [0, 1, 2, 3, 4],
    }
    variable_trend_y_ticks_dic = {
        'T_Ave': [0, 0.3, 0.6, 0.9, 1.2],
        'SR_Sum': [-300, -150, 0, 150, 300],
        'WS_Ave': [-0.2, -0.1, 0, 0.1, 0.2],
    }
    lw = 0.25


    for var_name in variables:
        plt.figure(figsize=(2.2, 1.5))
        y_ticks = variable_mean_y_ticks_dic[var_name]

        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            values = relation_df.loc[region_lake_ids, var_name].values

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
        plt.savefig(os.path.join(save_dir, 'Fig5 mete mean - {}.png'.format(var_name)), dpi=300., transparent=False)
        plt.cla()


        plt.figure(figsize=(2.2, 1.5))
        y_ticks = variable_trend_y_ticks_dic[var_name]

        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            values = relation_df.loc[region_lake_ids, '{}_trend'.format(var_name)].values

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
            if p <= p_value:
                y_loc = (y_ticks[0] + y_ticks[1]) * 0.5 \
                    if relation_df.loc[:, '{}_trend'.format(var_name)].values.mean() > y_ticks[2] \
                    else (y_ticks[-2] + y_ticks[-1]) * 0.5
                plt.scatter(
                    [region_id],
                    [y_loc],
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
        plt.savefig(os.path.join(save_dir, 'Fig5 mete trend - {}.png'.format(var_name)), dpi=300., transparent=False)
        plt.cla()


    # load variable value to shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(lake_point_shapefile_path, 1)

    layer = dataset.GetLayer()
    layer_defn = layer.GetLayerDefn()
    for var_name in variable_trend_y_ticks_dic.keys():
        mean_field_name, trd_field_name = var_name, 'trd_{}'.format(var_name)
        if layer_defn.GetFieldIndex(mean_field_name) < 0:
            new_float_field = ogr.FieldDefn(mean_field_name, ogr.OFTReal)
            layer.CreateField(new_float_field)
        if layer_defn.GetFieldIndex(trd_field_name) < 0:
            new_float_field = ogr.FieldDefn(trd_field_name, ogr.OFTReal)
            layer.CreateField(new_float_field)

    feature = layer.GetNextFeature()
    while feature:
        lake_id = str(feature.GetField('LakeID'))
        for var_name in variable_trend_y_ticks_dic.keys():
            feature.SetField('trd_{}'.format(var_name), relation_df.loc[lake_id, '{}_trend'.format(var_name)])
            feature.SetField(var_name, relation_df.loc[lake_id, var_name])
            layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    layer = None
    dataset = None


def fig6_variable_distribution():
    relation_df = pd.read_csv(os.path.join('Data', 'Lake mean attributes.csv'), index_col=0, header=0)
    relation_df['TN2TP'] = relation_df['TN'] / relation_df['TP']
    trend_df, significant_df = calc_mea_nutrient_trend()
    relation_df.loc[trend_df.index, trend_df.columns] = trend_df.values
    relation_df[['Chla', 'TP', 'TN']] *= 1000   # scale mg to ug
    relation_df[['Chla', 'TP', 'TN', 'TN2TP']] = np.log10(relation_df[['Chla', 'TP', 'TN', 'TN2TP']])
    relation_df['SR_Sum'] *= 365.25     # scale daily average to annual average
    for c in relation_df.columns:
        if str(c).endswith('trend'):
            relation_df[c] *= 10     # scale trend in year-1 to decade-1


    variable_label_dic = {
        'Chla': 'Chl$\it{a}$',
        'TP': 'TP',
        'TN': 'TN',
        'TN2TP': 'TN: TP',
        'T_Ave': 'Annual average T',
        'SR_Sum': 'Annual SR',
        'WS_Ave': 'Annual average WS',
    }
    log_variables = ['Chla', 'TP', 'TN', 'TN2TP']
    variable_unit_dic = {
        'Chla': 'mg L$^{-1}$',
        'TP': 'mg L$^{-1}$',
        'TN': 'mg L$^{-1}$',
        'TN2TP': '',
        'T_Ave': '$^o$C',
        'SR_Sum': 'MJ m$^{-2}$',
        'WS_Ave': 'm s$^{-1}$',
    }
    add_trend_unit = 'decade$^{-1}$'
    variable_mean_tick_dis_dic = {
        'Chla': 4,
        'TP': 2,
        'TN': 2,
        'TN2TP': 2,
        'T_Ave': 20,
        'SR_Sum': 2000,
        'WS_Ave': 2,
    }
    variable_trend_tick_dis_dic = {
        'Chla': 0.1,
        'TP': 0.1,
        'TN': 2,
        'TN2TP': 100,
        'T_Ave': 0.5,
        'SR_Sum': 200,
        'WS_Ave': 0.1,
    }
    lw = 0.25
    fs = 8


    fig, axes = plt.subplots(2, len(variable_label_dic), figsize=(len(variable_label_dic) * 2, 5))
    for i, (var_name, var_label) in enumerate(variable_label_dic.items()):
        plt.sca(axes[0, i])
        ax = plt.gca()
        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            values = relation_df.loc[region_lake_ids, var_name].values
            x_tick_max = (np.abs(values).max() // variable_mean_tick_dis_dic[var_name]) * variable_mean_tick_dis_dic[var_name] + variable_mean_tick_dis_dic[var_name]

            kde = stats.gaussian_kde(values)
            xx = np.linspace(-x_tick_max, x_tick_max, 1000)
            if region_id == 0:
                plt.fill_between(
                    xx,
                    np.zeros_like(xx),
                    kde(xx),
                    color=region_colors[region_id],
                    facecolor=region_colors[region_id],
                    alpha=1
                )
                plt.plot(xx, kde(xx), c='k', lw=lw, alpha=1)

        values = relation_df.loc[:, var_name].values
        x_tick_max = (np.abs(values).max() // variable_mean_tick_dis_dic[var_name]) * variable_mean_tick_dis_dic[var_name] + variable_mean_tick_dis_dic[var_name]
        if var_name not in log_variables:
            x_label = var_label + ' ({})'.format(variable_unit_dic[var_name])
        else:
            if var_name in log_variables[:-1]:
                x_label = 'log ' + var_label + ' × 1000 ({})'.format(variable_unit_dic[var_name])
            else:
                x_label = 'log ' + var_label + ' ({})'.format(variable_unit_dic[var_name])
        ax.set_xlabel(x_label, fontsize=fs)
        plt.xticks([0, x_tick_max * 0.25, x_tick_max * 0.5, x_tick_max * 0.75, x_tick_max])
        plt.xlim([0, x_tick_max])
        ax.tick_params(axis='both', labelsize=fs, width=lw, length=3)
        plt.grid(axis='both', color='grey', linestyle='--', linewidth=0.25, alpha=0.4)
        plt.text(0.025 if values.mean() > x_tick_max * 0.5 else 0.45, 0.8, 'Mean: {:.4f}'.format(np.mean(values)), transform=ax.transAxes, fontsize=fs - 1)
        plt.text(0.025 if values.mean() > x_tick_max * 0.5 else 0.45, 0.7, 'SD: {:.4f}'.format(np.std(values)), transform=ax.transAxes, fontsize=fs - 1)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        if i == 0:
            ax.set_ylabel('Density', fontsize=fs)

        plt.sca(axes[1, i])
        ax = plt.gca()
        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            values = relation_df.loc[region_lake_ids, '{}_trend'.format(var_name)].values
            x_tick_max = (np.abs(values).max() // variable_trend_tick_dis_dic[var_name]) * variable_trend_tick_dis_dic[var_name] + variable_trend_tick_dis_dic[var_name]

            kde = stats.gaussian_kde(values)
            xx = np.linspace(-x_tick_max, x_tick_max, 1000)
            if region_id == 0:
                plt.fill_between(
                    xx,
                    np.zeros_like(xx),
                    kde(xx),
                    color=region_colors[region_id],
                    facecolor=region_colors[region_id],
                    alpha=1
                )
                plt.plot(xx, kde(xx), c='k', lw=lw, alpha=1)

        values = relation_df.loc[:, '{}_trend'.format(var_name)].values
        x_tick_max = (np.abs(values).max() // variable_trend_tick_dis_dic[var_name]) * variable_trend_tick_dis_dic[var_name] + variable_trend_tick_dis_dic[var_name]
        ax.set_xlabel(var_label + ' trend \n ({} {})'.format(variable_unit_dic[var_name], add_trend_unit), fontsize=fs)
        plt.xticks([-x_tick_max, -x_tick_max * 0.5, 0, x_tick_max * 0.5, x_tick_max])
        plt.xlim([-x_tick_max, x_tick_max])
        ax.tick_params(axis='both', labelsize=fs, width=lw, length=3)
        plt.grid(axis='both', color='grey', linestyle='--', linewidth=0.25, alpha=0.4)
        plt.text(0.025 if values.mean() > 0 else 0.45, 0.8, 'Mean: {:.4f}'.format(np.mean(values)), transform=ax.transAxes, fontsize=fs - 1)
        plt.text(0.025 if values.mean() > 0 else 0.45, 0.7, 'SD: {:.4f}'.format(np.std(values)), transform=ax.transAxes, fontsize=fs - 1)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        if i == 0:
            ax.set_ylabel('Density', fontsize=fs)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig6 variable dist.png'), dpi=300.)


def fig7_variable_relationship():
    relation_df = pd.read_csv(os.path.join('Data', 'Lake mean attributes.csv'), index_col=0, header=0)
    relation_df['TN2TP'] = relation_df['TN'] / relation_df['TP']
    trend_df, significant_df = calc_mea_nutrient_trend()
    relation_df.loc[trend_df.index, trend_df.columns] = trend_df.values
    relation_df[['Chla', 'TP', 'TN']] *= 1000  # scale mg to ug
    relation_df[['Chla', 'TP', 'TN', 'TN2TP']] = np.log10(relation_df[['Chla', 'TP', 'TN', 'TN2TP']])
    relation_df['SR_Sum'] *= 365.25  # scale daily average to annual average
    for c in relation_df.columns:
        if str(c).endswith('trend'):
            relation_df[c] *= 10  # scale trend in year-1 to decade-1
    region_colors = ['#d3d3d3', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']
    print(relation_df['TN2TP_trend'])


    variable_label_dic = {
        'Chla': 'Chl$\it{a}$',
        'TP': 'TP',
        'TN': 'TN',
        'TN2TP': 'TN: TP',
        'T_Ave': 'Annual average T',
        'SR_Sum': 'Annual SR',
        'WS_Ave': 'Annual average WS',
    }
    log_variables = ['Chla', 'TP', 'TN', 'TN2TP']
    variable_unit_dic = {
        'Chla': 'mg L$^{-1}$',
        'TP': 'mg L$^{-1}$',
        'TN': 'mg L$^{-1}$',
        'TN2TP': '',
        'T_Ave': '$^o$C',
        'SR_Sum': 'MJ m$^{-2}$',
        'WS_Ave': 'm s$^{-1}$',
    }
    add_trend_unit = 'decade$^{-1}$'
    variable_mean_x_ticks_dic = {
        'Chla': [0, 0.5, 1, 1.5, 2],
        'TP': [0.5, 1, 1.5, 2, 2.5],
        'TN': [2, 2.5, 3, 3.5, 4],
        'TN2TP': [0.5, 1, 1.5, 2, 2.5],
        'T_Ave': [0, 7.5, 15, 22.5, 30],
        'SR_Sum': [3000, 4000, 5000, 6000, 7000],
        'WS_Ave': [0, 1.5, 3, 4.5, 6],
    }
    variable_trend_tick_dis_dic = {
        'Chla': 0.1,
        'TP': 0.1,
        'TN': 2,
        'TN2TP': 100,
        'T_Ave': 0.5,
        'SR_Sum': 200,
        'WS_Ave': 0.1,
    }
    y_ticks = [-0.12, -0.06, 0, 0.06, 0.12]
    lw = 0.25
    fs = 8


    fig, axes = plt.subplots(2, len(variable_label_dic), figsize=(len(variable_label_dic) * 2, 5), sharey='row')
    for i, (var_name, var_label) in enumerate(variable_label_dic.items()):
        plt.sca(axes[0, i])
        ax = plt.gca()
        for region_id in range(1, 5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            x_values = relation_df.loc[region_lake_ids, var_name].values
            y_values = relation_df.loc[region_lake_ids, 'Chla_trend'].values
            plt.scatter(
                x_values,
                y_values,
                marker='o',
                label=RegionName_dic[region_id - 1],
                s=10,
                linewidths=0.25,
                facecolors=region_colors[region_id],
                edgecolors='k',
                alpha=1
            )

        x_values = relation_df.loc[:, var_name].values
        y_values = relation_df.loc[:, 'Chla_trend'].values
        r, p = stats.pearsonr(x_values, y_values)
        slope, intercept, r_value, p_line, std_err = linregress(x_values, y_values)
        sns.regplot(
            x=x_values,
            y=y_values,
            ci=None,
            color='r' if slope > 0 else 'b',
            scatter=False,
            line_kws={'linewidth': 1, 'linestyle': '--' if p > p_value else '-', 'alpha': 1},
        )

        if var_name not in log_variables:
            x_label = var_label + ' ({})'.format(variable_unit_dic[var_name])
        else:
            if var_name in log_variables[:-1]:
                x_label = 'log ' + var_label + ' × 1000 ({})'.format(variable_unit_dic[var_name])
            else:
                x_label = 'log ' + var_label + ' ({})'.format(variable_unit_dic[var_name])
        ax.set_xlabel(x_label, fontsize=fs)
        plt.xticks(variable_mean_x_ticks_dic[var_name])
        plt.xlim([variable_mean_x_ticks_dic[var_name][0], variable_mean_x_ticks_dic[var_name][-1]])
        plt.yticks(y_ticks)
        plt.ylim([y_ticks[0], y_ticks[-1]])
        ax.tick_params(axis='both', labelsize=fs, width=lw, length=3)
        plt.grid(axis='both', color='grey', linestyle='--', linewidth=0.25, alpha=0.4)
        if p < p_value:
            plt.text(0.2, 0.1, 'R = {:.2f} (P < 0.1)'.format(r), transform=ax.transAxes, fontsize=fs)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        if i == 0:
            ax.set_ylabel('Chl$\it{a}$ trend (mg L$^{-1}$ decade$^{-1}$)', fontsize=fs)


        plt.sca(axes[1, i])
        ax = plt.gca()
        for region_id in range(1, 5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            x_values = relation_df.loc[region_lake_ids, '{}_trend'.format(var_name)].values
            y_values = relation_df.loc[region_lake_ids, 'Chla_trend'].values
            plt.scatter(
                x_values if i > 0 else [],
                y_values if i > 0 else [],
                marker='o',
                label=RegionName_dic[region_id - 1],
                s=10,
                linewidths=0.25,
                facecolors=region_colors[region_id],
                edgecolors='k',
                alpha=1
            )

        x_values = relation_df.loc[:, '{}_trend'.format(var_name)].values
        y_values = relation_df.loc[:, 'Chla_trend'].values
        r, p = stats.pearsonr(x_values, y_values)
        slope, intercept, r_value, p_line, std_err = linregress(x_values, y_values)
        if i > 0:
            sns.regplot(
                x=x_values,
                y=y_values,
                ci=None,
                color='r' if slope > 0 else 'b',
                scatter=False,
                line_kws={'linewidth': 1, 'linestyle': '--' if p > p_value else '-', 'alpha': 1},
            )

        x_tick_max = (np.abs(x_values).max() // variable_trend_tick_dis_dic[var_name]) * variable_trend_tick_dis_dic[var_name] + variable_trend_tick_dis_dic[var_name]
        ax.set_xlabel(var_label + ' trend \n ({} {})'.format(variable_unit_dic[var_name], add_trend_unit), fontsize=fs)
        plt.xticks([-x_tick_max, -x_tick_max * 0.5, 0, x_tick_max * 0.5, x_tick_max])
        plt.xlim([-x_tick_max, x_tick_max])
        plt.yticks(y_ticks)
        plt.ylim([y_ticks[0], y_ticks[-1]])
        ax.tick_params(axis='both', labelsize=fs, width=lw, length=3)
        plt.grid(axis='both', color='grey', linestyle='--', linewidth=0.25, alpha=0.4)
        if p < p_value and i > 0:
            plt.text(0.2, 0.1, 'R = {:.2f} (P < 0.1)'.format(r), transform=ax.transAxes, fontsize=fs)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        if i == 0:
            ax.set_ylabel('Chl$\it{a}$ trend (mg L$^{-1}$ decade$^{-1}$)', fontsize=fs)
            plt.legend(ncol=1, fontsize=fs, frameon=False, loc=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig7 variable relationship.png'), dpi=300.)


def fig8_zonal_meteorological_difference():
    model = Model(
        input_dir='Data/ModelInput',
        case_ids=list(UsedCases_dic.keys())
    )
    variable_y_tick_bound_dic = {
        'T_Ave': [-20, 40],
        'SR_Sum': [0, 30],
        'WS_Ave': [0, 6],
    }
    variable_y_label_dic = {
        'T_Ave': 'Average temperature\n($^o$C)',
        'SR_Sum': 'Solar radiation\n(MJ m$^{-2}$ d$^{-1}$)',
        'WS_Ave': 'Average wind speed\n(m s$^{-1}$)',
    }
    lw = 0.25


    iv_xr = model.Data['InputVars']
    data_xr = iv_xr.groupby('Date.dayofyear').mean('Date').rename({'dayofyear': 'Date'})


    fig, axes = plt.subplots(len(variable_y_tick_bound_dic), 1, figsize=(10, 1.5 * len(variable_y_tick_bound_dic)), sharex=True)
    for i, (var_name, (y_min, y_max)) in enumerate(variable_y_tick_bound_dic.items()):
        ax = axes[i]
        plt.sca(ax)

        for region_id, region_name in RegionName_dic.items():
            region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]

            region_df = data_xr.loc[region_case_ids, :, var_name].to_pandas().T
            region_mean_ss = region_df.mean(axis=1)
            region_q10_ss = region_df.apply(lambda x: x.quantile(0.1), axis=1)
            region_q90_ss = region_df.apply(lambda x: x.quantile(0.9), axis=1)

            ax.fill_between(
                region_q10_ss.index,
                region_q10_ss.values,
                region_q90_ss.values,
                facecolor=region_colors[region_id + 1],
                alpha=0.2
            )
            plt.plot(
                region_mean_ss.index,
                region_mean_ss.values,
                c=region_colors[region_id + 1],
                linewidth=lw * 3,
                alpha=1,
                linestyle='-',
                label=RegionName_dic[region_id]
            )

        plt.xlim([1, 365])
        plt.xticks([1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
        plt.ylim([y_min, y_max])
        slice = (y_max - y_min) / 3
        if i == 0:
            plt.yticks([y_min, y_min + slice, y_min + slice * 2, y_max])
        else:
            plt.yticks([y_min, y_min + slice, y_min + slice * 2])
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=lw, alpha=0.5)
        ax.tick_params(axis='both', labelsize=8, width=lw, length=3)
        ax.set_ylabel(variable_y_label_dic[var_name], fontsize=8)
        if i == 0:
            plt.legend(ncol=4, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(save_dir, 'Fig8 Seasonal differences of zonal mete.png'), dpi=300.)


if __name__ == '__main__':
    fig1to4_nutrient_trend_box()
    fig5_zonal_climate_change_box()
    fig6_variable_distribution()
    fig7_variable_relationship()
    fig8_zonal_meteorological_difference()



