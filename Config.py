import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, inspect
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.sans-serif'] = ['Arial']



F_types = ['Fb', 'Fd', 'Fg']
N_types = ['ON', 'NOx', 'NH3']
P_types = ['OP', 'PO4']
MeaTypes = ['Chla', 'TN', 'TP']
Initial_distribute_dics = {
    'TN': {'ON': 0.7, 'NH3': 0.2, 'NOx': 0.1},
    'TP': {'OP': 0.75, 'PO4': 0.25},
    'Chla': {'Fb': 0.5 / 0.02, 'Fd': 0.25 / 0.02, 'Fg': 0.25 / 0.02}
}
Merged_class_names = ['TF', 'TN', 'TP']


# Get lake information
Cases_df = pd.read_excel('Data/Model.xlsx', sheet_name='Cases', header=0, index_col='Index')
AllCases_dic = {lake_id: lake_name for lake_id, lake_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name', :]))}
UsedCases_dic = {lake_id: lake_name for lake_id, lake_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name', :])) if int(Cases_df.loc['Use', lake_id]) == 1}

AllRegions_dic = {lake_id: region_id for lake_id, region_id in zip(list(Cases_df.columns), list(Cases_df.loc['Region_ID', :]))}
RegionName_dic = {0: 'EPL', 1: 'YGPL', 2: 'NPML', 3: 'IMXL'}
RegionColor_dic = {0: 'r', 1: 'g', 2: '#e6b43c', 3: 'b'}

AllCasesLonLat_dic = {lake_id: [lon, lat] for lake_id, lon, lat in zip(list(Cases_df.columns), list(Cases_df.loc['Lon', :]), list(Cases_df.loc['Lat', :]))}
AllCasesEngName_dic = {lake_id: Eng_name for lake_id, Eng_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name_English', :]))}
AllCasesArea_dic = {lake_id: float(area * 1e-6) for lake_id, area in zip(list(Cases_df.columns), list(Cases_df.loc['Area', :]))}
RegionUseSumArea_dic = {region_id: np.sum([area for lake_id, area in AllCasesArea_dic.items() if AllRegions_dic[lake_id] == region_id]) for region_id in RegionName_dic.keys()}
AllCasesWaterDepth_dic = {lake_id: float(water_depth) for lake_id, water_depth in zip(list(Cases_df.columns), list(Cases_df.loc['WL', :]))}

# to stats flux dynamics [Geochemical >> biology >> management]
FluxInDic = {
    'Fb': ['Grow_Fb', 'Mix_Fb'],
    'Fd': ['Grow_Fd', 'Mix_Fd'],
    'Fg': ['Grow_Fg', 'Mix_Fg'],
    'TF': ['FromFb_TF', 'FromFd_TF', 'FromFg_TF'],

    'ON': ['Resuspension_ON', 'BiologyMetabolism_ON', 'Mix_ON'],
    'NH3': ['Mineralization_ON', 'Releasing_NH3', 'Deposition_NH3', 'BiologyMetabolism_NH3', 'Mix_NH3'],
    'NOx': ['Nitrification_NH3', 'Deposition_NOx', 'Releasing_NOx', 'BiologyMetabolism_NOx', 'Mix_NOx'],
    'TN': ['FromON_TN', 'FromNH3_TN', 'FromNOx_TN'],

    'OP': ['Resuspension_OP', 'BiologyMetabolism_OP', 'Mix_OP'],
    'PO4': ['Releasing_PO4', 'Mineralization_OP', 'BiologyMetabolism_PO4', 'Mix_PO4'],
    'TP': ['FromOP_TP', 'FromPO4_TP'],

    'WL': ['Inflow_WL', 'Precipitation_WL'],
    'DO': ['BiologyPhotosynthesis_DO', 'Deposition_DO', 'Mix_DO'],
    'Chla': ['InFb_Chla', 'InFd_Chla', 'InFg_Chla'],
}
FluxOutDic = {
    'Fb': ['Mortality_Fb', 'Excretion_Fb', 'Predation_Fb', 'Settling_Fb'],
    'Fd': ['Mortality_Fd', 'Excretion_Fd', 'Predation_Fd', 'Settling_Fd'],
    'Fg': ['Mortality_Fg', 'Excretion_Fg', 'Predation_Fg', 'Settling_Fg'],
    'TF': [],

    'ON': ['Mineralization_ON', 'Settling_ON'],
    'NH3': ['Nitrification_NH3', 'Anammox_NH3', 'BiologyUptake_NH3', 'Adsorption_NH3'],
    'NOx': ['Denitrification_NOx', 'BiologyUptake_NOx', 'Adsorption_NOx'],
    'TN': [],

    'OP': ['Mineralization_OP', 'Settling_OP'],
    'PO4': ['BiologyUptake_PO4', 'Adsorption_PO4'],
    'TP': [],

    'WL': ['Evaporation_WL', 'Seepage_WL', 'Outflow_WL'],
    'DO': ['BiologyRespiration_DO'],
    'Chla': [],
}


class Empty:
    def __init__(self, attr_dic=None):
        attr_dic = {} if isinstance(attr_dic, dict) else attr_dic
        _ = [setattr(self, key, value) for key, value in attr_dic.items()]
