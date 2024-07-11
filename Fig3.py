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
region_colors = ['#d3d3d3', '#bccd9a', '#f3c5d6', '#a2c7c8', '#fdf0bc']
save_dir = 'Results/Fig3'
lake_point_shapefile_path = 'Data/GeoData/Shapefiles/Lake_point.shp'


def load_lake_mean_risk():
    lake_mean_risk_df = pd.read_csv(os.path.join('Results/Fig3/Risk', '2005-2022 Lake mean risk.csv'), header=0, index_col=0)
    return lake_mean_risk_df


def lighten_color(color, amount=1.0):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def load_regional_daily_risk():
    regional_daily_risk_nr_df = pd.read_csv(os.path.join('Results/Fig3/Risk', '2005-2022 Zonal daily risk (Nutrient reduction).csv'), header=0, index_col=0)
    regional_daily_risk_nr_df.index = pd.to_datetime(regional_daily_risk_nr_df.index)
    regional_daily_risk_wnr_df = pd.read_csv(os.path.join('Results/Fig3/Risk', '2005-2022 Zonal daily risk (Without nutrient reduction).csv'), header=0, index_col=0)
    regional_daily_risk_wnr_df.index = pd.to_datetime(regional_daily_risk_wnr_df.index)
    return regional_daily_risk_nr_df, regional_daily_risk_wnr_df


def plt_regional_risk_box():
    lake_mean_risk_df = load_lake_mean_risk()
    lw = 0.25

    for i, col in enumerate(lake_mean_risk_df.columns):
        plt.figure(figsize=(2.2, 1.5))
        for region_id in range(5):

            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]

            values = lake_mean_risk_df.loc[region_lake_ids, col].values
            values += 0.0001    # must > 0, as y log scaled

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
        plt.xlim([-0.5, 4.5])
        plt.yscale('log')
        plt.ylim([1e-4, 3e0])
        plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], ['0.01%', '0.1%', '1%', '10%', '100%'])
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        plt.tight_layout()
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=lw, alpha=0.4)
        plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.97)
        plt.savefig(os.path.join(save_dir, 'Map sub box fig - {}.png'.format(col)), dpi=300., transparent=False)
        plt.cla()


def plt_regional_risk_bar():
    lake_mean_risk_df = load_lake_mean_risk()
    lw = 0.25


    risk_class_dic = {
        '#2b83ba': [-1, 0],
        '#ffffbf': [0, 0.01],
        '#fdae61': [0.01, 0.1],
        '#d7191c': [0.1, 999],
    }

    for col in lake_mean_risk_df.columns:
        percentage_df = pd.DataFrame(index=risk_class_dic.keys(), columns=range(5))
        count_df = pd.DataFrame(index=risk_class_dic.keys(), columns=range(5))
        plt.figure(figsize=(2.2, 1.5))
        for region_id in range(5):
            if region_id == 0:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys()]
            else:
                region_lake_ids = [lake_id for lake_id in UsedCases_dic.keys() if AllRegions_dic[lake_id] == region_id - 1]

            values = lake_mean_risk_df.loc[region_lake_ids, col].values

            sum_percentage = 0
            for risk_color, risk_bound in risk_class_dic.items():
                count = np.count_nonzero((values > risk_bound[0]) & (values <= risk_bound[1]))
                percentage = (count / len(values))
                plt.bar([region_id], [percentage], width=0.5, color=risk_color, bottom=sum_percentage, edgecolor='k', linewidth=lw)
                sum_percentage = sum_percentage + percentage
                percentage_df.loc[risk_color, region_id] = percentage
                count_df.loc[risk_color, region_id] = count

        ax = plt.gca()
        ax.xaxis.tick_top()
        plt.xlim([-0.5, 4.5])
        region_names = list(RegionName_dic.values())
        plt.xticks(range(0, len(region_names) + 1), ['All'] + list(RegionName_dic.values()))
        ax.yaxis.tick_right()
        plt.ylim([0, 1])
        plt.yticks([0.25, 0.5, 0.75, 1], ['25%', '50%', '75%', '100%'])
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=9)
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=lw)
        plt.tight_layout()
        plt.subplots_adjust(left=0.0, bottom=0.0, right=0.82, top=0.85)
        plt.savefig(os.path.join(save_dir, 'Map sub bar fig - {}.png'.format(col)), dpi=300., transparent=True)
        plt.cla()
        print(col, 'Percentage\n', percentage_df)
        print(col, 'Count\n', count_df)


def plt_regional_daily_risk_trend():
    regional_daily_risk_nr_df, regional_daily_risk_wnr_df = load_regional_daily_risk()
    region_yticks_dic = {
        0: [0, 0.05, 0.1, 0.15, 0.2],
        1: [0, 0.2, 0.4, 0.6, 0.8],
        2: [0, 0.01, 0.02, 0.03, 0.04],
        3: [0, 0.01, 0.02, 0.03, 0.04],
        4: [0, 0.03, 0.06, 0.09, 0.12],
    }


    fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex='col')
    for region_id in range(5):
        plt.sca(axes[region_id])
        ax = plt.gca()
        col = RegionName_dic[region_id] if region_id < 4 else 'Nation'
        regional_daily_risk_nr_ss = regional_daily_risk_nr_df[col]
        regional_daily_risk_wnr_ss = regional_daily_risk_wnr_df[col]

        regional_daily_risk_nr_year_mean = regional_daily_risk_nr_ss.groupby(regional_daily_risk_nr_ss.index.year).mean().values
        trend_slope = mk.sens_slope(regional_daily_risk_nr_year_mean)[0]
        trend_test = mk.original_test(regional_daily_risk_nr_year_mean, alpha=p_value)
        print(region_id, 'Nutrient reduction', 'trend: {} \t slope: {:.5f}'.format(trend_test.trend, trend_slope * 10.))

        regional_daily_risk_wnr_year_mean = regional_daily_risk_wnr_ss.groupby(regional_daily_risk_wnr_ss.index.year).mean().values
        trend_slope = mk.sens_slope(regional_daily_risk_wnr_year_mean)[0]
        trend_test = mk.original_test(regional_daily_risk_wnr_year_mean, alpha=p_value)
        print(region_id, 'Without nutrient reduction', 'trend: {} \t slope: {:.5f}'.format(trend_test.trend, trend_slope * 10.))

        print(region_id, 'Nutrient reduction mean: {:.5f} \t Without nutrient reduction mean: {:.5f} \t Ratio: {:.5f}'
              .format(regional_daily_risk_nr_ss.mean(),
                      regional_daily_risk_wnr_ss.mean(),
                      regional_daily_risk_nr_ss.mean() / regional_daily_risk_wnr_ss.mean()
                      )
              )

        # Fit GAM model to make curve smoother
        regional_daily_risk_nr_gam_ss = deepcopy(regional_daily_risk_nr_ss)
        regional_daily_risk_wnr_gam_ss = deepcopy(regional_daily_risk_wnr_ss)
        gam = LinearGAM(n_splines=200, spline_order=4).fit(range(len(regional_daily_risk_nr_gam_ss.index)), regional_daily_risk_nr_gam_ss.values)
        regional_daily_risk_nr_gam_ss.loc[:] = gam.predict(range(len(regional_daily_risk_nr_gam_ss.index)))
        gam = LinearGAM(n_splines=200, spline_order=4).fit(range(len(regional_daily_risk_wnr_gam_ss.index)), regional_daily_risk_wnr_gam_ss.values)
        regional_daily_risk_wnr_gam_ss.loc[:] = gam.predict(range(len(regional_daily_risk_wnr_gam_ss.index)))

        ax.fill_between(
            regional_daily_risk_nr_gam_ss.index,
            [0] * len(regional_daily_risk_nr_gam_ss.values),
            regional_daily_risk_nr_gam_ss.values,
            color='none',
            facecolor=region_colors[region_id + 1 if region_id < 4 else 0],
            alpha=0.75,
            interpolate=True,
            linewidth=1,
        )
        # plt.plot(ini_mean_gam_ss.index, ini_mean_gam_ss.values, lw=1, c=sce_colors[region_id])

        ax.fill_between(
            regional_daily_risk_nr_gam_ss.index,
            regional_daily_risk_nr_gam_ss.values,
            regional_daily_risk_wnr_gam_ss.values,
            color='none',
            facecolor='#5db3cb',
            alpha=0.75,
            interpolate=True,
            linewidth=1,
        )

        x_year_mean = np.array([(i + 1) * 182.75 for i in range(len(regional_daily_risk_nr_year_mean))])
        x_year_date = pd.to_datetime(['{}-9-1'.format(year) for year in range(2005, 2023)])
        # plt.scatter(
        #     x_year_date,
        #     regional_daily_risk_nr_year_mean,
        #     marker='o',
        #     s=20,
        #     linewidths=0.5,
        #     facecolors=region_colors[region_id + 1 if region_id < 4 else 0],
        #     edgecolors='k',
        #     alpha=1,
        # )
        # plt.scatter(
        #     x_year_date,
        #     regional_daily_risk_wnr_year_mean,
        #     marker='o',
        #     s=20,
        #     linewidths=0.5,
        #     facecolors='#5db3cb',  # 'none'
        #     edgecolors='k',
        #     alpha=1,
        # )

        slope, intercept, r_value, p, std_err = linregress(x_year_mean, regional_daily_risk_nr_year_mean)
        plt.plot(
            x_year_date,
            x_year_mean * slope + intercept,
            c=region_colors[region_id + 1 if region_id < 4 else 0],
            lw=2,
            alpha=1,
            ls='--',
            label=' ',
        )
        slope, intercept, r_value, p, std_err = linregress(x_year_mean, regional_daily_risk_wnr_year_mean)
        plt.plot(
            x_year_date,
            x_year_mean * slope + intercept,
            c='#5db3cb',
            lw=2,
            alpha=1,
            ls='--',
            label=' ',
        )

        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xlim([regional_daily_risk_nr_ss.index[0], pd.to_datetime('2023-1-1')])

        plt.yticks(region_yticks_dic[region_id])
        plt.ylim([region_yticks_dic[region_id][0], region_yticks_dic[region_id][-1]])

        plt.grid(axis='both', which='major', color='grey', linestyle='--', linewidth=0.25, alpha=0.25)

        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
        ax.tick_params(axis='both', which='both', labelsize=8, width=0.25, length=3)
        # plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Regional risk trend.png'), dpi=300.)
    plt.cla()


def plt_regional_yearly_risk_bar():
    regional_daily_risk_nr_df, regional_daily_risk_wnr_df = load_regional_daily_risk()
    bar_width = 1


    fig, axes = plt.subplots(1, 5, figsize=(8, 2), sharex='col', sharey='all')
    axes = axes.flatten()
    # fig.delaxes(axes[-1])
    # plt.figure(figsize=(8, 4))
    for region_id in range(5):
        plt.sca(axes[region_id - 1 if region_id > 0 else 4])
        ax = plt.gca()

        col = RegionName_dic[region_id - 1] if region_id > 0 else 'Nation'
        regional_daily_risk_nr_ss = regional_daily_risk_nr_df[col]
        regional_daily_risk_wnr_ss = regional_daily_risk_wnr_df[col]

        regional_daily_risk_nr_year_mean = regional_daily_risk_nr_ss.groupby(regional_daily_risk_nr_ss.index.year).mean().values
        trend_slope = mk.sens_slope(regional_daily_risk_nr_year_mean)[0]
        trend_test = mk.original_test(regional_daily_risk_nr_year_mean, alpha=p_value)
        print(region_id, 'Nutrient reduction', 'trend: {} \t slope: {:.5f}'.format(trend_test.trend, trend_slope * 10.))

        regional_daily_risk_wnr_year_mean = regional_daily_risk_wnr_ss.groupby(regional_daily_risk_wnr_ss.index.year).mean().values
        trend_slope = mk.sens_slope(regional_daily_risk_wnr_year_mean)[0]
        trend_test = mk.original_test(regional_daily_risk_wnr_year_mean, alpha=p_value)
        print(region_id, 'Without nutrient reduction', 'trend: {} \t slope: {:.5f}'.format(trend_test.trend, trend_slope * 10.))

        print(region_id, 'Nutrient reduction mean: {:.5f} \t Without nutrient reduction mean: {:.5f} \t Ratio: {:.5f}'
              .format(regional_daily_risk_nr_ss.mean(),
                      regional_daily_risk_wnr_ss.mean(),
                      regional_daily_risk_nr_ss.mean() / regional_daily_risk_wnr_ss.mean()
                      )
              )

        plt.bar(
            np.arange(2005, 2023), regional_daily_risk_nr_year_mean,
            width=bar_width,
            color=region_colors[region_id],
            edgecolor='k', linewidth=0.25,
        )
        plt.bar(
            np.arange(2005, 2023), regional_daily_risk_wnr_year_mean - regional_daily_risk_nr_year_mean,
            width=bar_width,
            color='#5db3cb',
            bottom=regional_daily_risk_nr_year_mean,
            edgecolor='k', linewidth=0.25,
        )

        plt.plot(
            [2005, 2009], [regional_daily_risk_nr_year_mean[:5].mean(), regional_daily_risk_nr_year_mean[:5].mean()],
            c=lighten_color(region_colors[region_id], 1.8),
            lw=1,
            alpha=1,
            ls='--',
        )
        plt.plot(
            [2005, 2009], [regional_daily_risk_wnr_year_mean[:5].mean(), regional_daily_risk_wnr_year_mean[:5].mean()],
            c=lighten_color('#5db3cb', 1.5),
            lw=1,
            alpha=1,
            ls='--',
        )

        plt.plot(
            [2018, 2022], [regional_daily_risk_nr_year_mean[-5:].mean(), regional_daily_risk_nr_year_mean[-5:].mean()],
            c=lighten_color(region_colors[region_id], 1.8),
            lw=1,
            alpha=1,
            ls='--',
            label='{:.2f}% to {:.2f}%'.format(regional_daily_risk_nr_year_mean[:5].mean() * 100, regional_daily_risk_nr_year_mean[-5:].mean() * 100)
        )
        plt.plot(
            [2018, 2022], [regional_daily_risk_wnr_year_mean[-5:].mean(), regional_daily_risk_wnr_year_mean[-5:].mean()],
            c=lighten_color('#5db3cb', 1.5),
            lw=1,
            alpha=1,
            ls='--',
            label='{:.2f}% to {:.2f}%'.format(regional_daily_risk_wnr_year_mean[:5].mean() * 100, regional_daily_risk_wnr_year_mean[-5:].mean() * 100)
        )

        print(region_id, 'HRI ratio: {:.2f}'.format((1 - (regional_daily_risk_nr_year_mean[-5:].mean() / regional_daily_risk_wnr_year_mean[-5:].mean())) * 100))

        plt.yscale('log')
        plt.ylim([1e-4, 3e0])
        plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], ['0.01%', '0.1%', '1%', '10%', '100%'])
        plt.xticks([2005, 2010, 2015, 2020], [2005, 2010, 2015, 2020])
        plt.xlim([2005-bar_width*0.5, 2022+bar_width*0.5])

        plt.text(0.025, 0.92, col, transform=plt.gca().transAxes, fontsize=8)

        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        # ax.xaxis.set_major_locator(mdates.YearLocator(5))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.grid(axis='y', which='major', color='grey', linestyle='--', linewidth=0.25, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
        ax.tick_params(axis='both', which='both', labelsize=8, width=0.25, length=3)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        if region_id == 1:
            plt.ylabel('$HRI_{Z}$' + ' / ' + '$HRI_{N}$', fontsize=8)
        leg = plt.legend(ncol=1, fontsize=6, loc=1, frameon=False)
        leg.get_frame().set_linewidth(0.0)

    plt.subplots_adjust(wspace=-0.5, hspace=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Regional yearly risk bar.png'), dpi=300.)
    plt.cla()


def load_lake_risk_to_map():
    lake_mean_risk_df = load_lake_mean_risk()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(lake_point_shapefile_path, 1)
    layer = dataset.GetLayer()
    layer_defn = layer.GetLayerDefn()

    col_field_pair_dic = {
        'Nutrient reduction': 'r_pst_nr',
        'Without nutrient reduction': 'r_pst_wnr',
    }
    for name in col_field_pair_dic.values():
        if layer_defn.GetFieldIndex(name) < 0:
            new_field = ogr.FieldDefn(name, ogr.OFTReal)
            layer.CreateField(new_field)

    feature = layer.GetNextFeature()
    while feature:
        lake_id = str(feature.GetField('LakeID'))
        for col, field in col_field_pair_dic.items():
            feature.SetField(field, lake_mean_risk_df.loc[lake_id, col])
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()

    layer = None
    dataset = None


if __name__ == '__main__':
    # load_lake_risk_to_map()
    plt_regional_risk_box()
    # plt_regional_risk_bar()
    # plt_regional_yearly_risk_bar()
    # plt_regional_daily_risk_trend()



