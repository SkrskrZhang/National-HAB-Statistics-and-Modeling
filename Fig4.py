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
from Fig3 import lighten_color
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


p_value = 0.1
Ns = ['TP', 'TN', 'TN2TP', 'Chla']
mea_data_dir = 'Data/MeaData'
region_colors = ['lightgrey', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']
save_dir = 'Results/Fig4'
lake_point_shapefile_path = 'Data/GeoData/Shapefiles/Lake_point.shp'


def load_regional_future_daily_risk(zone_name):
    regional_daily_risk_df = pd.read_csv(os.path.join('Results/Fig4/Risk', '{} future risk.csv'.format(zone_name)), header=0, index_col=0)
    regional_daily_risk_df.index = pd.to_datetime(regional_daily_risk_df.index)
    return regional_daily_risk_df


def plt_regional_yearly_risk_bar():
    bar_width = 1


    fig, axes = plt.subplots(2, 4, figsize=(8, 4), sharex='col', sharey='all')
    # axes = axes.flatten()
    # fig.delaxes(axes[-1])
    # plt.figure(figsize=(8, 4))
    for i_row, mete_sce in enumerate(['ssp245', 'ssp585']):
        for region_id in range(4):
            plt.sca(axes[i_row, region_id])
            ax = plt.gca()

            regional_daily_risk_df = load_regional_future_daily_risk(RegionName_dic[region_id])
            regional_yearly_risk_df = regional_daily_risk_df.groupby(regional_daily_risk_df.index.year).mean()

            regional_yearly_risk_nr_ss = regional_yearly_risk_df['{} + {}'.format(mete_sce, 'nutrient decrease')].loc[2025:]
            regional_yearly_risk_wnr_ss = regional_yearly_risk_df['{} + {}'.format(mete_sce, 'nutrient nodecrease')].loc[2025:]

            plt.bar(
                np.arange(2025, 2051), regional_yearly_risk_nr_ss.values,
                width=bar_width,
                color=region_colors[region_id+1],
                edgecolor='k', linewidth=0.25,
            )
            plt.bar(
                np.arange(2025, 2051), regional_yearly_risk_wnr_ss.values - regional_yearly_risk_nr_ss.values,
                width=bar_width,
                color='#5db3cb',
                bottom=regional_yearly_risk_nr_ss.values,
                edgecolor='k', linewidth=0.25,
            )

            plt.plot(
                [2025, 2029], [regional_yearly_risk_nr_ss.values[:5].mean(), regional_yearly_risk_nr_ss.values[:5].mean()],
                c=lighten_color(region_colors[region_id + 1], 1.8),
                lw=1,
                alpha=1,
                ls='--',
                # label='{:.2f}%'.format(regional_yearly_risk_nr_ss.values[:5].mean() * 100)
            )
            plt.plot(
                [2025, 2029], [regional_yearly_risk_wnr_ss.values[:5].mean(), regional_yearly_risk_wnr_ss.values[:5].mean()],
                c=lighten_color('#5db3cb', 1.5),
                lw=1,
                alpha=1,
                ls='--',
                # label='{:.2f}%'.format(regional_yearly_risk_wnr_ss.values[:5].mean() * 100)
            )

            plt.plot(
                [2046, 2050], [regional_yearly_risk_nr_ss.values[-5:].mean(), regional_yearly_risk_nr_ss.values[-5:].mean()],
                c=lighten_color(region_colors[region_id + 1], 1.8),
                lw=1,
                alpha=1,
                ls='--',
                label='{:.2f}% to {:.2f}%'.format(regional_yearly_risk_nr_ss.values[:5].mean() * 100, regional_yearly_risk_nr_ss.values[-5:].mean() * 100)
            )
            plt.plot(
                [2046, 2050], [regional_yearly_risk_wnr_ss.values[-5:].mean(), regional_yearly_risk_wnr_ss.values[-5:].mean()],
                c=lighten_color('#5db3cb', 1.5),
                lw=1,
                alpha=1,
                ls='--',
                label='{:.2f}% to {:.2f}%'.format(regional_yearly_risk_wnr_ss.values[:5].mean() * 100, regional_yearly_risk_wnr_ss.values[-5:].mean() * 100)
            )

            print(region_id, mete_sce, 'reduction:{:.4f}\tno reduction:{:.4f}'.format(regional_yearly_risk_nr_ss.values[-5:].mean() / regional_yearly_risk_nr_ss.values[:5].mean(), regional_yearly_risk_wnr_ss.values[-5:].mean() / regional_yearly_risk_wnr_ss.values[:5].mean()))


            plt.yscale('log')
            plt.ylim([1e-4, 3e0])
            plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], ['0.01%', '0.1%', '1%', '10%', '100%'])
            plt.xticks([2025, 2030, 2035, 2040, 2045, 2050])
            plt.xlim([2025-bar_width*0.5, 2050+bar_width*0.5])

            plt.text(0.025, 0.92, RegionName_dic[region_id], transform=plt.gca().transAxes, fontsize=8)

            # ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            # ax.xaxis.set_major_locator(mdates.YearLocator(5))
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.grid(axis='y', which='major', color='grey', linestyle='--', linewidth=0.25, alpha=0.25)
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)
            ax.tick_params(axis='both', which='both', labelsize=7, width=0.25, length=3)
            params = {'mathtext.default': 'regular'}
            plt.rcParams.update(params)
            if region_id == 0:
                plt.ylabel('$HRI_{Z}$' + ' in ' + mete_sce.replace('ssp', 'SSP').replace('245', '2-4.5').replace('585', '5-8.5'), fontsize=8)
            leg = plt.legend(ncol=1, fontsize=7, loc=1, frameon=False)
            leg.get_frame().set_linewidth(0.0)

    plt.subplots_adjust(wspace=-0.5, hspace=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Regional yearly risk bar.png'), dpi=300.)
    plt.cla()


if __name__ == '__main__':
    # load_lake_risk_to_map()
    # plt_regional_risk_box()
    plt_regional_yearly_risk_bar()
    # plt_regional_yearly_risk_bar()
    # plt_regional_daily_risk_trend()



