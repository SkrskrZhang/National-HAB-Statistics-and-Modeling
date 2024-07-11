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
from matplotlib.ticker import ScalarFormatter

from Config import *
from Fig1 import calc_mea_nutrient_trend
from NationalHABModel import Model
from ModelFit import ModelFit
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


p_value = 0.1
Ns = ['Chla', 'TP', 'TN', 'TN2TP']
mea_data_dir = 'Data/MeaData'
region_colors = ['lightgrey', '#50730a', '#b70046', '#006c6e', '#bd9700']
save_dir = 'Results/Figs in Supplementary Information'
lake_point_shapefile_path = 'Data/GeoData/Shapefiles/Lake_point.shp'


def fig1_studied_lakes_attr():
    nutrients = ['Chla', 'TP', 'TN', 'TN2TP']
    meteorology = ['T_Ave', 'SR_Sum', 'WS_Ave']
    geographical = ['Area', 'Depth']
    y_max_ls = [0.1, 0.2, 6, 200, 30, 7000, 6, 5, 3]
    y_min_ls = [0, 0, 0, 0, 0, 3000, 0, 0, 0]
    lw = 0.25
    y_tick_labels = [
        'Chlorophyll $\it{a}$ (mg L$^{-1}$)',
        'Total phosphorous (mg L$^{-1}$)',
        'Total nitrogen (mg L$^{-1}$)',
        'Total nitrogen: Total phosphorous',
        'Annual average temperature ($^o$C)',
        'Annual solar radiation (MJ m$^{-2}$)',
        'Annual average wind speed (m s$^{-1}$)',
        'log Surface area (km$^{2}$)',
        'log Average depth (m)',
    ]


    data_df = pd.read_csv('Data/Lake mean attributes.csv', header=0, index_col=0)
    data_df['TN2TP'] = data_df['TN'] / data_df['TP']
    data_df.loc[:, 'SR_Sum'] *= 365.25
    data_df.loc[:, 'Depth'] += 0.01
    data_df.loc[:, geographical] = np.log10(data_df.loc[:, geographical])
    data_df = data_df[nutrients + meteorology + geographical]


    fig, axes = plt.subplots(9, 1, figsize=(16, 8), sharex=True)
    axes = axes.flatten()


    for i_ax, column in enumerate(data_df.columns):
        ax = axes[i_ax]
        plt.sca(ax)

        i = 0
        i_names = []
        for region_id, region_name in RegionName_dic.items():
            region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]

            sort_region_data_df = data_df.loc[region_case_ids, :].sort_values(by='Chla', ascending=False)
            sort_region_data_ss = sort_region_data_df.loc[:, column]

            plt.bar(
                range(i, len(region_case_ids) + i),
                [y_max_ls[i_ax]] * len(region_case_ids),
                width=0.1,
                color=region_colors[region_id + 1],
                edgecolor='none',
                alpha=0.2
            )
            plt.bar(
                range(i, len(region_case_ids) + i),
                sort_region_data_ss.values,
                width=lw * 2,
                color=region_colors[region_id + 1],
                edgecolor='k',
                linewidth=lw,
                label=region_name if i_ax == 0 else None
            )

            i = i + len(region_case_ids)
            i_names.extend([AllCasesEngName_dic[case_id] for case_id in sort_region_data_ss.index])

        for ii, i_name in enumerate(i_names):
            i_names[ii] = i_name.replace('Lake', 'L.').replace('Reservoir', 'R.')
        plt.xticks(range(i), i_names, rotation=270)
        plt.xlim([-1, len(i_names)])

        for spine in ax.spines.values():
            spine.set_linewidth(lw)

        ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=8)

        plt.text(0.005, 0.8, y_tick_labels[i_ax], transform=ax.transAxes, fontsize=8)

        y_max, y_min = y_max_ls[i_ax], y_min_ls[i_ax]
        if i_ax == 0:
            plt.yticks([y_min, (y_max + y_min) / 2, y_max])
        else:
            plt.yticks([y_min, (y_max + y_min) / 2])
        plt.ylim([y_min, y_max])

        plt.grid(which='major', axis='y', color='grey', linestyle='--', linewidth=lw, alpha=0.5)

        if i_ax == 0:
            plt.legend(ncol=4, fontsize=8, loc=1, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(save_dir, 'Fig1 Lake attr.png'), dpi=300.)


def fig4_paras_sensitivity_box():
    from ModelParasSensitiveAanlysis import ModelParasSensitiveAnalysis

    repeat_onetime = 100
    n_repeat = 10
    load_dir = 'Results/ModelParasSensitiveAnalysis'
    save_path = os.path.join(save_dir, 'Fig4 Paras Sensitive Box.png')

    for start_repeat in [i * repeat_onetime for i in range(n_repeat)]:
        ModelParasSensitiveAnalysis.morris_train(
            start_repeat=start_repeat,
            save_dir=load_dir
        )
    ModelParasSensitiveAnalysis.morris_analysis(load_dir, range(repeat_onetime * n_repeat))

    ModelParasSensitiveAnalysis.plt_morris_results(load_dir, save_path)


def fig5_lake_inflow():
    lw = 0.25

    n_row, n_col = 25, 4
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 12), sharex=True)
    axes = axes.flatten()
    i = 0
    for region_id, region_name in RegionName_dic.items():
        region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]
        for case_id in region_case_ids:
            ax = axes[i]
            plt.sca(ax)

            formatter = ScalarFormatter(useMathText=True)
            plt.gca().get_yaxis().set_major_formatter(formatter)

            input_var_df = pd.read_csv(os.path.join('Data/ModelInput', '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'InputVars')), header=0, index_col=0)
            input_var_df.index = pd.to_datetime(input_var_df.index)

            area = AllCasesArea_dic[case_id] * 1e6
            inflow_ss = input_var_df.loc[:, 'W_Inflow'] * area * 1e-3

            plt.fill_between(
                inflow_ss.index,
                [0] * len(inflow_ss),
                inflow_ss.values,
                # color=scenario_color_dic[scenario] if np_type == 'decrease' else 'none',
                facecolor=region_colors[region_id + 1],
                alpha=0.8
            )

            plt.text(0.025, 0.75, AllCasesEngName_dic[case_id], transform=ax.transAxes, fontsize=6)


            plt.xlim([pd.to_datetime('2005-1-1'), pd.to_datetime('2023-1-1')])
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))

            n_pos = len(str(int(inflow_ss.mean())))
            unit = np.power(10, n_pos - 1)
            y_max = (inflow_ss.max() // unit) * unit + unit
            plt.ylim([0, y_max])
            if i < n_col:
                plt.yticks([0, y_max / 2, y_max])
            else:
                plt.yticks([0, y_max / 2])
            ax.tick_params(axis='y', labelsize=6)

            for spine in ax.spines.values():
                spine.set_linewidth(lw)
            ax.tick_params(axis='both', which='major', width=lw, length=4)
            ax.tick_params(axis='both', which='minor', width=lw, length=2)

            plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=lw, alpha=0.5)
            i += 1

    for ax in axes[-3:]:
        ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.25)
    plt.savefig(os.path.join(save_dir, 'Fig5 Lake Daily Inflow.png'), dpi=300.)


def fig6_HierarchicalBayesian_results():
    dir_optimization = 'Results/ModelOptimizationResults'
    optimized_paras_ls = [
        'Rate_Grow_Fb', 'Rate_Grow_Fd', 'Rate_Grow_Fg', 'HSC_P_Grow_Fb', 'Opt_T_Grow_Fb',
        'Rate_Settling_ON', 'Rate_Releasing_NH3', 'Rate_Denitrification_NOx',
        'Rate_Settling_OP', 'Rate_Releasing_PO4',
    ]


    # Run HierarchicalBayesian and save results
    from ModelOptimization import ModelOptimizerHierarchicalBayesian
    from multiprocessing import freeze_support
    freeze_support()
    optimizer = ModelOptimizerHierarchicalBayesian(
        case_ids=list(UsedCases_dic.keys()),
        target='Chla',
    )
    optimizer.hierarchical_bayes(
        n_chains=5,
        n_samples=5000,
        save_dir=dir_optimization,
    )
    ModelOptimizerHierarchicalBayesian.estimate_paras_distribution(dir_optimization)


    # Plt results
    optimized_paras_mean_df = pd.read_csv(os.path.join(dir_optimization, 'Last100_ParasMean.csv'), index_col=0, header=0)
    optimized_paras_std_df = pd.read_csv(os.path.join(dir_optimization, 'Last100_ParasStd.csv'), index_col=0, header=0)
    paras_yticks_dic = {
        'Rate_Grow_Fb': [0.2, 0.8],
        'Rate_Grow_Fd': [0.6, 1.8],
        'Rate_Grow_Fg': [0.5, 1.4],
        'HSC_P_Grow_Fb': [0.002, 0.011],
        'Opt_T_Grow_Fb': [27, 30],
        'Rate_Settling_ON': [0, 0.09],
        'Rate_Releasing_NH3': [0, 0.012],
        'Rate_Denitrification_NOx': [0, 0.12],
        'Rate_Settling_OP': [0, 0.12],
        'Rate_Releasing_PO4': [0, 0.006],
    }
    n_ticks = 3
    lw = 0.25

    plt.figure(figsize=(12, 15))
    for i_row, paras_name in enumerate(optimized_paras_ls):
        ax_nation = plt.subplot2grid((len(optimized_paras_ls), 10), (i_row, 0))
        plt.sca(ax_nation)
        plt.errorbar(
            [0], [optimized_paras_mean_df.loc[:, paras_name].mean()], yerr=[optimized_paras_mean_df.loc[:, paras_name].std()],
            fmt='o',
            ecolor='k',
            elinewidth=lw * 8,
            capsize=4,
            capthick=2,
            ms=8,
            mec='k',
            mfc='k',
        )
        plt.xticks([0], ['All'], rotation=270)
        plt.xlim([-1, 1])
        y_tick_step = (paras_yticks_dic[paras_name][1] - paras_yticks_dic[paras_name][0]) / n_ticks
        y_ticks = [paras_yticks_dic[paras_name][0] + y_tick_step * yi for yi in range(n_ticks + 1)]
        y_ticks = y_ticks if i_row == 0 else y_ticks[:-1]
        plt.yticks(y_ticks)
        plt.ylim([paras_yticks_dic[paras_name][0], paras_yticks_dic[paras_name][1]])
        ax_nation.tick_params(axis='both', labelsize=8)
        plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=lw, alpha=0.5)
        text_paras_name = paras_name.replace('Rate', 'K')\
            .replace('HSC_P', 'HKP')\
            .replace('Opt_T', 'OptT')\
            .replace('Resuspension', 'Resu')\
            .replace('Releasing', 'Rele')\
            .replace('Denitrification', 'Denit')\
            .replace('Settling', 'Settl')
        plt.text(0.025, 0.875, text_paras_name, transform=ax_nation.transAxes, fontsize=11)
        if i_row == 0:
            plt.text(0.01, 0.6, 'Higher hierarchy\n(All lakes)', transform=ax_nation.transAxes, fontsize=10)
        for spine in ax_nation.spines.values():
            spine.set_linewidth(lw)
        ax_nation.tick_params(axis='both', which='major', width=lw, length=4)
        ax_nation.tick_params(axis='both', which='minor', width=lw, length=2)


        ax_lake = plt.subplot2grid((len(optimized_paras_ls), 10), (i_row, 1), colspan=9)
        plt.sca(ax_lake)
        i = 0
        i_names = []
        for region_id, region_name in RegionName_dic.items():
            region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]
            means = optimized_paras_mean_df.loc[region_case_ids, paras_name].values
            means[np.isnan(means)] = np.nanmean(means)
            stds = optimized_paras_std_df.loc[region_case_ids, paras_name].values
            stds[np.isnan(stds)] = np.nanmean(stds)
            plt.errorbar(
                range(i, len(region_case_ids) + i), means, yerr=stds,
                fmt='o',
                ecolor=region_colors[region_id + 1],
                elinewidth=1,
                capsize=2,
                capthick=1,
                ms=4,
                mec=region_colors[region_id + 1],
                mfc=region_colors[region_id + 1],
                label=region_name if i_row == 0 else None
            )
            i += len(region_case_ids)
            i_names.extend([AllCasesEngName_dic[region_case_id] for region_case_id in region_case_ids])

        for ii, i_name in enumerate(i_names):
            i_names[ii] = i_name.replace('Lake', 'L.').replace('Reservoir', 'R.')
        plt.xticks(range(i), i_names, rotation=270)
        plt.xlim([-1, len(i_names)])
        plt.yticks([], [])
        plt.ylim([paras_yticks_dic[paras_name][0], paras_yticks_dic[paras_name][1]])
        ax_lake.tick_params(axis='both', labelsize=8)
        plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=0.25, alpha=0.5)
        if i_row == 0:
            plt.text(0.005, 0.7, 'Lower hierarchy (Individual lake)', transform=ax_lake.transAxes, fontsize=10)
        if i_row == 0:
            plt.legend(ncol=4, fontsize=10, loc=1, frameon=False)
        for spine in ax_lake.spines.values():
            spine.set_linewidth(lw)
        ax_lake.tick_params(axis='both', which='major', width=lw, length=4)
        ax_lake.tick_params(axis='both', which='minor', width=lw, length=2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(os.path.join(save_dir, 'Fig6 Paras Distribution.png'), dpi=300.)


def fig7_calibrated_model_phytoplankton_dynamics():
    # Calibrated model run
    from ModelOptimization import calibrated_national_phytoplankton_module_run
    dir_optimization = 'Results/ModelOptimizationResults'
    calibrated_national_phytoplankton_module_run(dir_optimization)


    # Plt phytoplankton dynamics according to the simulated biomass
    dir_calibrated_results = 'Results/CalibratedModelSimulationResults'
    f_color_dic = {
        'Fb': 'blue',
        'Fd': 'red',
        'Fg': 'green',
    }
    lw = 0.25

    n_row, n_col = 25, 4
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 12), sharex=True)
    axes = axes.flatten()
    i = -1
    for region_id, region_name in RegionName_dic.items():
        region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]
        for case_id in region_case_ids:
            i += 1
            ax = axes[i]
            plt.sca(ax)

            simulated_sv_df = pd.read_csv(os.path.join(dir_calibrated_results, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'StateVars', 0)), index_col=0, header=0)
            simulated_sv_df.index = pd.to_datetime(simulated_sv_df.index)

            base_arr = np.zeros(shape=(len(simulated_sv_df.index),), dtype=np.float32)
            for f_name, f_color in f_color_dic.items():
                plt.fill_between(
                    simulated_sv_df.index,
                    base_arr,
                    simulated_sv_df.loc[:, f_name].values + base_arr,
                    # color=scenario_color_dic[scenario] if np_type == 'decrease' else 'none',
                    facecolor=f_color,
                    alpha=1,
                )
                base_arr += simulated_sv_df.loc[:, f_name].values

            plt.text(0.025, 0.75, AllCasesEngName_dic[case_id], transform=ax.transAxes, fontsize=6, color=region_colors[AllRegions_dic[case_id] + 1])
            plt.xlim([pd.to_datetime('2005-1-1'), pd.to_datetime('2023-1-1')])
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            y_max = (np.nanmax(base_arr) // 0.01) * 0.01 + 0.01
            plt.ylim([0, y_max])
            if i < n_col:
                plt.yticks([0, y_max / 2, y_max])
            else:
                plt.yticks([0, y_max / 2])
            ax.tick_params(axis='y', labelsize=6)
            for spine in ax.spines.values():
                spine.set_linewidth(lw)
            ax.tick_params(axis='both', which='major', width=lw, length=4)
            ax.tick_params(axis='both', which='minor', width=lw, length=2)
            plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=lw, alpha=0.5)

    for ax in axes[-3:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.25)
    plt.savefig(os.path.join(save_dir, 'Fig7 Phytoplankton group dynamics.png'), dpi=300.)


def fig8_calibrated_model_chla_dynamics():
    # Calibrated model run
    from ModelOptimization import calibrated_national_phytoplankton_module_run
    dir_optimization = 'Results/ModelOptimizationResults'
    calibrated_national_phytoplankton_module_run(dir_optimization)


    # Plt phytoplankton dynamics according to the simulated biomass
    dir_calibrated_results = 'Results/CalibratedModelSimulationResults'
    lw = 0.25

    n_row, n_col = 25, 4
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 12), sharex=True)
    axes = axes.flatten()
    i = -1
    for region_id, region_name in RegionName_dic.items():
        region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]
        for case_id in region_case_ids:
            i += 1
            ax = axes[i]
            plt.sca(ax)

            simulated_sv_df = pd.read_csv(os.path.join(dir_calibrated_results, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'StateVars', 0)), index_col=0, header=0)
            simulated_sv_df.index = pd.to_datetime(simulated_sv_df.index)
            mea_chla_df = pd.read_csv(os.path.join('Data/MeaData', '{}-Chla-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_chla_df.index = pd.to_datetime(mea_chla_df.index)

            plt.plot(
                simulated_sv_df.index,
                simulated_sv_df['Chla'].values,
                linewidth=lw,
                c=region_colors[region_id + 1]
            )

            plt.scatter(
                mea_chla_df.index,
                mea_chla_df['Chla'].values,
                s=2,
                linewidths=lw * 0.5,
                facecolors='none',
                edgecolors='red'
            )

            nse = ModelFit.get_nse(mea_chla_df['Chla'].values, simulated_sv_df.loc[mea_chla_df.index, 'Chla'].values)

            plt.text(0.025, 0.75, AllCasesEngName_dic[case_id], transform=ax.transAxes, fontsize=6)
            plt.text(0.025, 0.45, 'NSE: {:.2f}'.format(nse), transform=ax.transAxes, fontsize=6)
            plt.xlim([pd.to_datetime('2005-1-1'), pd.to_datetime('2023-1-1')])
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            y_max = (simulated_sv_df['Chla'].max() // 0.01) * 0.01 + 0.01
            plt.ylim([0, y_max])
            if i < n_col:
                plt.yticks([0, y_max / 2, y_max])
            else:
                plt.yticks([0, y_max / 2])
            ax.tick_params(axis='y', labelsize=6)
            for spine in ax.spines.values():
                spine.set_linewidth(lw)
            ax.tick_params(axis='both', which='major', width=lw, length=4)
            ax.tick_params(axis='both', which='minor', width=lw, length=2)
            plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=lw, alpha=0.5)

    for ax in axes[-3:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.25)
    plt.savefig(os.path.join(save_dir, 'Fig8 Calibrated chla dynamics.png'), dpi=300.)


def fig9_scenario_chla_dynamics_without_nutrient_reduction():
    # Scenario model run
    from ScenarioAnalysis import ScenarioModelOfWithoutNutrientReductionDuring2005to2022
    dir_calibrated_paras = 'Results/ModelOptimizationResults'
    dir_scenario_results = 'Results/ScenarioModelResultsWithoutNutrientReduction'
    scenario_model = ScenarioModelOfWithoutNutrientReductionDuring2005to2022(
        dir_calibrated_paras=dir_calibrated_paras
    )
    scenario_model.run(
        save_dir=dir_scenario_results
    )


    # Plt scenario results
    dir_ini_results = 'Results/CalibratedModelSimulationResults'
    lw = 0.25
    n_row, n_col = 25, 4
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 12), sharex=True)
    axes = axes.flatten()
    i = -1
    for region_id, region_name in RegionName_dic.items():
        region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]
        for case_id in region_case_ids:
            i += 1
            ax = axes[i]
            plt.sca(ax)

            scenario_sv_df = pd.read_csv(os.path.join(dir_scenario_results, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'StateVars', 0)), index_col=0, header=0)
            scenario_sv_df.index = pd.to_datetime(scenario_sv_df.index)
            ini_sv_df = pd.read_csv(os.path.join(dir_ini_results, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'StateVars', 0)), index_col=0, header=0)
            ini_sv_df.index = pd.to_datetime(ini_sv_df.index)

            plt.fill_between(
                ini_sv_df.index,
                [0] * ini_sv_df.shape[0],
                ini_sv_df['Chla'].values,
                facecolor=region_colors[region_id + 1],
                alpha=0.75
            )

            plt.fill_between(
                scenario_sv_df.index,
                ini_sv_df['Chla'].values,
                scenario_sv_df['Chla'].values,
                facecolor='red',
                alpha=1
            )

            plt.text(0.025, 0.75, AllCasesEngName_dic[case_id], transform=ax.transAxes, fontsize=6)
            plt.xlim([pd.to_datetime('2005-1-1'), pd.to_datetime('2023-1-1')])
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            y_max = (scenario_sv_df['Chla'].max() // 0.01) * 0.01 + 0.01
            plt.ylim([0, y_max])
            if i < n_col:
                plt.yticks([0, y_max / 2, y_max])
            else:
                plt.yticks([0, y_max / 2])
            ax.tick_params(axis='y', labelsize=6)
            for spine in ax.spines.values():
                spine.set_linewidth(lw)
            ax.tick_params(axis='both', which='major', width=lw, length=4)
            ax.tick_params(axis='both', which='minor', width=lw, length=2)
            plt.grid(which='major', axis='both', color='grey', linestyle='--', linewidth=lw, alpha=0.5)

    for ax in axes[-3:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.25)
    plt.savefig(os.path.join(save_dir, 'Fig9 Scenario chla dynamics.png'), dpi=300.)


def fig10_load_future_climate_trend_to_map():
    future_climate_trend_df = pd.read_csv(os.path.join('Data', 'Lake future climate trend.csv'), index_col=0, header=0)
    future_climate_trend_df *= 10.  # trend with unit (.../year) to (.../decade)
    scenarios = ['ssp245', 'ssp585']
    variables = ['T_Ave', 'SR_Sum', 'WS_Ave']
    for scenario in scenarios:
        future_climate_trend_df['SR_Sum{}'.format(scenario)] *= 365.25


    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(lake_point_shapefile_path, 1)


    layer = dataset.GetLayer()
    layer_defn = layer.GetLayerDefn()
    for scenario in scenarios:
        for variable in variables:
            trd_field_name = 'trd_{}_{}'.format(variable[:2], scenario[-3:])
            if layer_defn.GetFieldIndex(trd_field_name) < 0:
                new_float_field = ogr.FieldDefn(trd_field_name, ogr.OFTReal)
                layer.CreateField(new_float_field)


    feature = layer.GetNextFeature()
    while feature:
        lake_id = str(feature.GetField('LakeID'))
        for scenario in scenarios:
            for variable in variables:
                trd_field_name = 'trd_{}_{}'.format(variable[:2], scenario[-3:])
                feature.SetField(trd_field_name, future_climate_trend_df.loc[lake_id, '{}{}'.format(variable, scenario)])
                layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    layer = None
    dataset = None


def fig10_load_future_climate_trend_box():
    future_climate_trend_df = pd.read_csv(os.path.join('Data', 'Lake future climate trend.csv'), index_col=0, header=0)
    future_climate_trend_df *= 10.  # trend with unit (.../year) to (.../decade)
    scenarios = ['ssp245', 'ssp585']
    variables = ['T_Ave', 'SR_Sum', 'WS_Ave']
    for scenario in scenarios:
        future_climate_trend_df['SR_Sum{}'.format(scenario)] *= 365.25

    variable_trend_y_ticks_dic = {
        'T_Ave': [0, 0.2, 0.4, 0.6, 0.8],
        'SR_Sum': [-80, -40, 0, 40, 80],
        'WS_Ave': [-0.06, -0.03, 0, 0.03, 0.06],
    }
    lw = 0.25
    region_colors = ['#d3d3d3', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']


    for scenario in scenarios:
        for var_name in variables:
            plt.figure(figsize=(2.2, 1.5))
            y_ticks = variable_trend_y_ticks_dic[var_name]

            for region_id in range(5):
                if region_id == 0:
                    region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
                else:
                    region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
                values = future_climate_trend_df.loc[region_lake_ids, '{}{}'.format(var_name, scenario)].values

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
                        if future_climate_trend_df.loc[:, '{}{}'.format(var_name, scenario)].values.mean() > y_ticks[2] \
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
            plt.savefig(os.path.join(save_dir, 'Fig10 future mete trend - {}{}.png'.format(var_name, scenario)), dpi=300., transparent=False)
            plt.cla()


if __name__ == '__main__':
    # fig1_studied_lakes_attr()
    # fig4_paras_sensitivity_box()
    # fig5_lake_inflow()
    # fig6_HierarchicalBayesian_results()
    # fig7_calibrated_model_phytoplankton_dynamics()
    # fig8_calibrated_model_chla_dynamics()
    # fig9_scenario_chla_dynamics_without_nutrient_reduction()
    # fig10_load_future_climate_trend_to_map()
    fig10_load_future_climate_trend_box()
