import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pymannkendall as mk
from osgeo import ogr
from copy import deepcopy
from pygam import LinearGAM, s, f
from scipy.stats import gaussian_kde
from scipy.stats import linregress
import matplotlib.dates as mdates

from Config import *
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


p_value = 0.1
Ns = ['TP', 'TN', 'TN2TP', 'Chla']
mea_data_dir = 'Data/MeaData'
region_colors = ['lightgrey', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']
save_dir = 'Results/Fig5'
lake_point_shapefile_path = 'Data/GeoData/Shapefiles/Lake_point.shp'


def load_nutrient_mean_values():
    case_ids = list(UsedCases_dic.keys())
    nutrient_mean_value_df = pd.DataFrame(index=case_ids)

    for case_id in case_ids:
        N_dic = {}
        for N in Ns:
            if N != 'TN2TP':
                N_data_path = os.path.join(mea_data_dir, '{}-{}-Mean.csv'.format(case_id, N))
                N_data_df = pd.read_csv(N_data_path, index_col=0, header=0)
                N_mean = N_data_df[N].mean()
                N_dic[N] = N_mean
            else:
                N_mean = N_dic['TN'] / N_dic['TP']
            nutrient_mean_value_df.loc[case_id, N] = N_mean

    return nutrient_mean_value_df


def plt_regional_nutrient_box():
    nutrient_mean_value_df = load_nutrient_mean_values()

    y_labels = [
        'Chl$\it{a}$ (mg L$^{-1}$)',
        'TP (mg L$^{-1}$)',
        'TN (mg L$^{-1}$)',
        'TN:TP',
    ]

    y_tick_dic = {
        "Chla": [0.0, 0.025, 0.05, 0.075, 0.1],
        "TP": [0.0, 0.05, 0.1, 0.15, 0.2],
        "TN": [0, 1.5, 3, 4.5, 6],
        "TN2TP": [0, 50, 100, 150, 200],
    }

    lw = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex='col')
    axes = axes.flatten()

    for i, (N, y_ticks) in enumerate(y_tick_dic.items()):
        plt.sca(axes[i])

        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]
            values = nutrient_mean_value_df.loc[region_lake_ids, N].values

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
                s=15,
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
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=lw, alpha=0.4)
        plt.ylabel(y_labels[i], fontsize=8)
        plt.tight_layout()
    # plt.subplots_adjust(left=0.18, bottom=0.15, right=0.98, top=0.97)
    plt.savefig(os.path.join(save_dir, 'Nutrient box.png'), dpi=300., transparent=False)
    plt.cla()


def load_is_lake_to_map():
    from osgeo import ogr

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(lake_point_shapefile_path, 1)

    flag_name = 'IsLake'

    layer = dataset.GetLayer()
    layer_defn = layer.GetLayerDefn()
    new_int_field = ogr.FieldDefn(flag_name, ogr.OFTInteger)
    if layer_defn.GetFieldIndex(flag_name) < 0:
        layer.CreateField(new_int_field)

    feature = layer.GetNextFeature()
    while feature:
        lake_id = str(feature.GetField('LakeID'))
        if 'Lake' in AllCasesEngName_dic[lake_id]:
            feature.SetField(flag_name, 1)
        else:
            feature.SetField(flag_name, 0)
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    layer = None
    dataset = None


if __name__ == '__main__':
    plt_regional_nutrient_box()
    # load_is_lake_to_map()



