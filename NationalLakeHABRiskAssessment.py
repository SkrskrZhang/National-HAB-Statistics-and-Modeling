import copy
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import linregress
from brokenaxes import brokenaxes
import matplotlib.dates as mdates
from pygam import LinearGAM, s, f
from scipy.stats import gaussian_kde
import pymannkendall as mk
import tqdm
import warnings

from Config import *


warnings.filterwarnings("ignore")
matplotlib.use('Agg')


class NationalHABRisk:
    def __init__(self):
        pass

    @staticmethod
    def calc_risk(load_dir, save_dir, case_ids, repeat=0):
        chl_threshold = 0.012
        max_ws = 3

        sv_df = pd.read_csv(os.path.join(load_dir, '{}-{}-StateVars-{}.csv'.format(case_ids[0], AllCases_dic[case_ids[0]], repeat)), index_col='Date', header=0)
        sv_df.index = pd.to_datetime(sv_df.index)
        case_risk_df = pd.DataFrame(index=sv_df.index, columns=case_ids)
        for case_id in case_ids:
            sv_df = pd.read_csv(os.path.join(load_dir, '{}-{}-StateVars-{}.csv'.format(case_id, AllCases_dic[case_id], repeat)), index_col='Date', header=0)
            sv_df.index = pd.to_datetime(sv_df.index)
            iv_df = pd.read_csv(os.path.join(load_dir, '{}-{}-InputVars-{}.csv'.format(case_id, AllCases_dic[case_id], repeat)), index_col='Date', header=0)
            iv_df.index = pd.to_datetime(iv_df.index)

            risks = sv_df['Chla'].values / chl_threshold
            risks = np.where(iv_df['WS_Ave'].values > max_ws, 0, risks)
            case_risk_df.loc[:, case_id] = risks

        case_risk_df.to_csv(os.path.join(save_dir, 'Lake HAB risk.csv'), index_label='Date')

        return case_risk_df

    @staticmethod
    def aggr_region_risk(case_risk_df, save_dir):
        """
        From daily risk calc the area-avg region risk
        """
        region_risk_df = copy.deepcopy(case_risk_df)
        for case_id in case_risk_df.columns:
            area_ratio = AllCasesArea_dic[case_id] / RegionUseSumArea_dic[AllRegions_dic[case_id]]
            region_risk_df.loc[:, case_id] = case_risk_df.loc[:, case_id] * area_ratio

        for region_id, region_name in RegionName_dic.items():
            region_case_ids = [case_id for case_id in UsedCases_dic.keys() if AllRegions_dic[case_id] == region_id]
            region_risk_df[region_name] = region_risk_df[region_case_ids].sum(axis=1)

        region_risk_df = region_risk_df[RegionName_dic.values()]
        region_risk_df.to_csv(os.path.join(save_dir, 'Zonal HAB risk.csv'), index_label='Date')

        return region_risk_df

    @staticmethod
    def aggr_country_risk(case_risk_df, save_dir):
        """
        From daily risk calc the area-avg country risk
        """
        country_risk_df = copy.deepcopy(case_risk_df)
        total_country_area = np.sum(list(RegionUseSumArea_dic.values()))
        for case_id in case_risk_df.columns:
            area_ratio = AllCasesArea_dic[case_id] / total_country_area
            country_risk_df.loc[:, case_id] = case_risk_df.loc[:, case_id] * area_ratio

        country_risk_df['All'] = country_risk_df[UsedCases_dic.keys()].sum(axis=1)
        country_risk_df = country_risk_df[['All']]
        country_risk_df.to_csv(os.path.join(save_dir, 'National HAB risk.csv'), index_label='Date')

        return country_risk_df


if __name__ == '__main__':
    load_dir = 'Results/ModelOptimizationResults'
    save_dir = 'Results/NationalHABRisk'

    case_risk_df = NationalHABRisk.calc_risk(
        load_dir=load_dir,
        save_dir=save_dir,
        case_ids=UsedCases_dic.keys()
    )

    NationalHABRisk.aggr_region_risk(
        case_risk_df,
        save_dir=save_dir,
    )

    NationalHABRisk.aggr_country_risk(
        case_risk_df,
        save_dir=save_dir,
    )
