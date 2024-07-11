import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import time
from copy import deepcopy
from functools import reduce

from Config import *
from WaterBalance.WaterBalance import LakeWaterBalance
from BioGeoChemistry.Nitrogen import LakeNitrogen
from BioGeoChemistry.Phosphorus import LakePhosphorus
from BioGeoChemistry.Phytoplankton import LakePhytoplankton
from BioGeoChemistry.DissolvedOxygen import LakeDissolvedOxygen
from BioGeoChemistry.Limitations.TLimitation import TemperatureLimitation
from BioGeoChemistry.Limitations.OLimitation import OxygenLimitation
from BioGeoChemistry.Limitations.NLimitation import NitrogenLimitation
from BioGeoChemistry.Limitations.PLimitation import PhosphorusLimitation
from BioGeoChemistry.Limitations.LLimitation import LightLimitation
from Utils.MutilProcess import mutil_process


class Model:
    def __init__(self, input_dir, case_ids, n_repeat=1, start_repeat=0, **kwargs):
        # Load model attr / data
        self.case_ids = case_ids if isinstance(case_ids, list) else [case_ids]
        self.input_dir = input_dir
        self.n_repeat = int(n_repeat)
        self.start_repeat = start_repeat
        self.repeat_range = range(self.start_repeat, self.start_repeat + self.n_repeat)


        # How to deal the SV out in range
        self.deal_range_out = 'break' if 'deal_range_out' not in kwargs.keys() else kwargs['deal_range_out']
        self.if_print_range_out = True if 'if_print_range_out' not in kwargs.keys() else kwargs['if_print_range_out']

        # Load model data
        self.Data = None
        self.union_date_index = None
        self.case_run_date_dic = None
        self.date_range_dic = None
        if 'Data' not in kwargs.keys() or ('Data' in kwargs.keys() and kwargs['Data'] is None):
            load_data_types = ['StateVars', 'Paras', 'Flux'] if 'load_data_types' not in kwargs.keys() else kwargs['load_data_types']
            self.load_data(load_data_types=load_data_types)
        else:
            self.Data = kwargs['Data']

        # Set the defined Date
        if self.union_date_index is not None and self.case_run_date_dic is not None:
            self.crt_date, self.last_date = self.union_date_index[1], self.union_date_index[0]
            self.start_date, self.stop_date = self.union_date_index[0], self.union_date_index[-1]
            self.crt_date_run_cases = self.case_run_date_dic[self.crt_date]

        self.load_modules()

    def run(self, if_show_bar=None, **kwargs):
        if_show_bar = False if if_show_bar is None else bool(if_show_bar)

        # set run date range
        self.start_date = self.union_date_index[0] if 'start_date' not in kwargs.keys() else pd.to_datetime(kwargs['start_date'])
        self.stop_date = self.union_date_index[-1] if 'start_date' not in kwargs.keys() else pd.to_datetime(kwargs['stop_date'])
        run_date_range = self.union_date_index[self.union_date_index.get_loc(self.start_date): self.union_date_index.get_loc(self.stop_date) + 1]

        # modeling step by step
        phar = None if not if_show_bar else tqdm(
            total=len(self.union_date_index) - 1,
            ncols=150,
            desc='RUN Lakes:{} ({})\tDaterange:{} to {}\t{} steps'
            .format(len(self.case_ids), '{}-{}'.format(self.case_ids[0], self.case_ids[-1]), self.start_date.strftime('%Y-%m-%d'), self.stop_date.strftime('%Y-%m-%d'), len(self.union_date_index) - 1)
        )

        for crt_date in run_date_range:
            self.run_step(crt_date)
            if self.check_range():
                if self.deal_range_out == 'clip':
                    print('Model run break because SV value not in check range, the variable value has been clipped.')
                elif self.deal_range_out == 'break':
                    print('Model run break because SV value not in check range, modeling break.')
                    break

            _ = None if not if_show_bar else phar.update()
        _ = None if not if_show_bar else phar.close()

    def run_step(self, date=None, **kwargs):
        """
        Run model one step.
        Process:
            1. Set model current/last date
            2. Load changeable paras to Date['Paras'] from Date['ParasVars'] if need
            3. Run WL module
            4. Run F N P module
                a) Run F, N and P modules
                b) Check union flux of NH3, NOx and PO4, because F uptake them in the min ratio
                c) Check all flux in F, N and P except NH3, NOx and PO4
            5. Run DO dynamics
            6. Set cruuent changed paras to Date['ParasVars'] from Date['Paras'] if need
        """


        # run model one step
        self.crt_date = self.crt_date if not date else pd.to_datetime(date)
        self.last_date = self.union_date_index[self.union_date_index.get_loc(self.crt_date) - 1]
        self.crt_date_run_cases = self.case_run_date_dic[self.crt_date]


        """-----------------------WL module-----------------------"""
        # calc W_WL fluxes OUT: (Atm, Seep, Outflow) IN: (Atm Inflow), Now, keep WL constant
        self.WaterBalance.dynamics()
        self.update_sv('WL')
        self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, 'Outflow_WL'] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, 'WL'] - self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, 'WL']
        self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, 'WL'] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, 'WL']
        """-----------------------WL module-----------------------"""


        """-----------------------F NP biogeochemical(BGC) cycle module-----------------------"""
        # F Meta fluxes OUT: (mortality, excretion, predation, settling) IN: (Growth)
        self.Phytoplankton.dynamics()
        self.Nitrogen.dynamics()
        self.Phosphorus.dynamics()

        # Check output sum < Available, check F
        min_ratios = self.check_flux(['NH3', 'NOx', 'PO4'], union=True)
        for component in self.Phytoplankton.components:
            self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, 'Grow_{}'.format(component)] = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, 'Grow_{}'.format(component)] * min_ratios
        self.check_flux(['ON', 'OP'])
        self.check_flux(self.Phytoplankton.components)
        self.update_sv(self.Phytoplankton.components)
        self.update_sv(self.Nitrogen.components)
        self.update_sv(self.Phosphorus.components)

        # Update Chla
        self.Phytoplankton.derive()

        # Update TN/TP/TF
        self.Nitrogen.aggregate()
        self.Phosphorus.aggregate()
        self.Phytoplankton.aggregate()
        self.update_sv(Merged_class_names)
        # self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, 'TF'] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, 'Fb'] + self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, 'Fd']
        """-----------------------F NP biogeochemical(BGC) cycle module-----------------------"""


        """-----------------------DO module-----------------------"""
        self.DissolvedOxygen.dynamics()
        self.update_sv('DO')
        """-----------------------DO module-----------------------"""

    def update_sv(self, sv_names):
        if isinstance(sv_names, str):
            sv_name = sv_names
            self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values + self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxInDic[sv_name]].values.sum(axis=2)
            self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name].values - self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxOutDic[sv_name]].values.sum(axis=2)
        elif isinstance(sv_names, list):
            for sv_name in sv_names:
                self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values + self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxInDic[sv_name]].values.sum(axis=2)
                self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name].values - self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxOutDic[sv_name]].values.sum(axis=2)

    def check_range(self, bias=0.001, **kwargs):
        p = False
        _ = np.zeros(shape=(len(self.crt_date_run_cases), self.n_repeat))
        for sv_name in FluxInDic.keys():
            if sv_name in self.Data['RangeStateVars'].coords['Index'].values:
                sv_values = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name]
                range_min, range_max = self.Data['RangeStateVars'].loc[self.crt_date_run_cases, sv_name, 'RangeMin'].values.reshape(-1, 1), self.Data['RangeStateVars'].loc[self.crt_date_run_cases, sv_name, 'RangeMax'].values.reshape(-1, 1)
                _ = np.where(sv_values < range_min, 1, 0)
                _ = np.where(sv_values > range_max, 1, _)

                # print information of out range
                if _.any():
                    if self.deal_range_out == 'clip':
                        self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name] = self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.crt_date, sv_name].clip(range_min + bias, range_max - bias)

                    if self.if_print_range_out:
                        for i in range(_.shape[0]):
                            for j in range(_.shape[1]):
                                if _[i, j] > 0:
                                    print('\nSV value out range find in case {} repeat {}\t date:{}\tname:{}\tvalue:{:.6f}'.format(self.crt_date_run_cases[i], j+self.start_repeat, self.crt_date, sv_name, self.Data['StateVars'].loc[self.crt_date_run_cases[i], j+self.start_repeat, self.crt_date, sv_name].values))
                                    sss = print(self.Data['Flux'].loc[self.crt_date_run_cases[i], j+self.start_repeat, self.crt_date, FluxInDic[sv_name] + FluxOutDic[sv_name]].to_pandas()) if sv_name in FluxInDic.keys() else None
                                    p = True
        return p

    def check_flux(self, sv_names, union=False, ratio_max_flux=0.85):
        """
        change outputs, avoid sv out flux
            if str  -> single min ratio
            if list -> coupled min ratio if union == True, else single min ratio in for
        """
        min_ratios = np.ones(shape=(len(self.crt_date_run_cases), self.n_repeat), dtype=np.float32)
        if isinstance(sv_names, str):
            sv_name = sv_names
            output_sum = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxOutDic[sv_name]].values.sum(axis=2)
            min_ratios = np.where(output_sum > (self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values * ratio_max_flux), (self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values * ratio_max_flux) / output_sum, min_ratios)
            if (min_ratios < 1).any():
                for flux_out in FluxOutDic[sv_name]:
                    self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, flux_out] = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, flux_out].values * min_ratios

        elif isinstance(sv_names, list):
            if union:
                union_min_ratios = np.ones(shape=(len(self.crt_date_run_cases), self.n_repeat), dtype=np.float32)
                for sv_name in sv_names:
                    min_ratios = np.ones(shape=(len(self.crt_date_run_cases), self.n_repeat), dtype=np.float32)
                    output_sum = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxOutDic[sv_name]].values.sum(axis=2)
                    min_ratios = np.where(output_sum > (self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values * ratio_max_flux), (self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values * ratio_max_flux) / output_sum, min_ratios)
                    union_min_ratios = np.where(union_min_ratios > min_ratios, min_ratios, union_min_ratios)
                if (union_min_ratios < 1).any():
                    for sv_name in sv_names:
                        for flux_out in FluxOutDic[sv_name]:
                            self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, flux_out] = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, flux_out].values * union_min_ratios
                min_ratios = union_min_ratios
            else:
                for sv_name in sv_names:
                    min_ratios[:, :] = 1
                    output_sum = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, FluxOutDic[sv_name]].values.sum(axis=2)
                    min_ratios = np.where(output_sum > (self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values * ratio_max_flux), (self.Data['StateVars'].loc[self.crt_date_run_cases, :, self.last_date, sv_name].values * ratio_max_flux) / output_sum, min_ratios)
                    if (min_ratios < 1).any():
                        for flux_out in FluxOutDic[sv_name]:
                            self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, flux_out] = self.Data['Flux'].loc[self.crt_date_run_cases, :, self.crt_date, flux_out].values * min_ratios
        return min_ratios

    def load_data(self, load_dir=None, if_show_bar=False, load_data_types=None, load_repeats=None):
        """
        First, Load the sv data of every case and get the union data index and every case run data dic (the run first data of a case is its date index[1])
        Load data, the no repeat data will be load anyway if not exists (repeat data will load its ini DF, like Paras_ini)
        The repeat data will load according to the load_data_types

        load_repeats:
            list: length must like self.case_ids, Then set to all cases with the pair data. [[[from repeat idx ls], [to repeat idx ls]], [], ]
            dic: like {case_id: [[from repeat idx ls], [to repeat idx ls]]}.
            None: load all repeat with the same repeat index
                The value in list or dic could be int or list
        """
        print('Load model input data...')
        flux_columns = [flux_in for sv_name, flux_ins in FluxInDic.items() for flux_in in flux_ins]
        _ = [flux_columns.append(flux_out) for sv_name, flux_outs in FluxOutDic.items() for flux_out in flux_outs]
        flux_columns = list(set(flux_columns))
        flux_columns.sort()  # MUST, as set() is not unique

        # The StateVars_ini / Paras_ini / IV / IC / RSV / MD will only be loaded once in any time
        if self.Data is None:
            self.Data = {}
            self.date_range_dic = {}
            sv_ini_dic = {}
            paras_ini_dic = {}
            rangeParas_ini_dic = {}
            meaData_columns_ls = []
            for case_id in self.case_ids:
                case_sv_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'StateVars')), header=0)
                case_sv_df = case_sv_df.set_index(pd.to_datetime(case_sv_df['Date'])).drop('Date', axis=1)
                sv_ini_dic[case_id] = case_sv_df
                self.date_range_dic[case_id] = case_sv_df.index
                case_paras_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'Paras')), header=0, index_col='Index')
                paras_ini_dic[case_id] = case_paras_df
                rangeParas_ini_dic[case_id] = case_paras_df[['Calibration', 'RangeMin', 'RangeMax']]

                case_md_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'MeaData')), header=0)
                case_md_df = case_md_df.set_index(pd.to_datetime(case_md_df['Date'])).drop('Date', axis=1)
                meaData_columns_ls.append(case_md_df.columns)
            self.Data['StateVars_ini'] = sv_ini_dic
            self.Data['Paras_ini'] = paras_ini_dic
            self.Data['RangeParas_ini'] = rangeParas_ini_dic


            # Get union date index for all cases, and get the run date case index
            self.union_date_index = reduce(lambda x, y: x.union(y), list(self.date_range_dic.values()))
            self.case_run_date_dic = {date: [] for date in self.union_date_index}
            for date in self.union_date_index:
                for case_id, case_date_range in self.date_range_dic.items():
                    if date in case_date_range[1:]:
                        self.case_run_date_dic[date].append(case_id)
            all_mea_columns = reduce(lambda x, y: x.union(y), meaData_columns_ls)

            # Load IV, MD, RangeSV, IC, RangeParas
            case0_iv_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(self.case_ids[0], AllCases_dic[self.case_ids[0]], 'InputVars')), header=0)
            case0_iv_df = case0_iv_df.set_index(pd.to_datetime(case0_iv_df['Date'])).drop('Date', axis=1)
            iv_xr = xr.DataArray(
                    np.full((len(self.case_ids), len(self.union_date_index), case0_iv_df.shape[1]), np.nan, dtype=np.float32),
                    dims=("CaseID", "Date", "Name"),
                    coords={
                        "CaseID": self.case_ids,
                        "Date": self.union_date_index,
                        "Name": list(case0_iv_df.columns),
                    },
            )
            md_xr = pd.DataFrame(
                index=self.union_date_index,
                columns=list(all_mea_columns),
            )
            case0_rsv_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(self.case_ids[0], AllCases_dic[self.case_ids[0]], 'RangeStateVars')), header=0, index_col='Index')
            rsv_xr = xr.DataArray(
                np.full((len(self.case_ids), case0_rsv_df.shape[0], case0_rsv_df.shape[1]), np.nan, dtype=np.float32),
                dims=("CaseID", "Index", "Name"),
                coords={
                    "CaseID": self.case_ids,
                    "Index": case0_rsv_df.index,
                    "Name": list(case0_rsv_df.columns),
                },
            )
            case0_ic_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(self.case_ids[0], AllCases_dic[self.case_ids[0]], 'InputCons')), header=0, index_col='Index')
            ic_xr = xr.DataArray(
                np.full((len(self.case_ids), case0_ic_df.shape[0], case0_ic_df.shape[1]), np.nan, dtype=np.float32),
                dims=("CaseID", "Index", "Name"),
                coords={
                    "CaseID": self.case_ids,
                    "Index": case0_ic_df.index,
                    "Name": list(case0_ic_df.columns),
                },
            )
            case0_rp_df = self.Data['RangeParas_ini'][self.case_ids[0]]
            rp_xr = xr.DataArray(
                np.full((len(self.case_ids), case0_rp_df.shape[0], case0_rp_df.shape[1]), np.nan, dtype=np.float32),
                dims=("CaseID", "Index", "Name"),
                coords={
                    "CaseID": self.case_ids,
                    "Index": case0_rp_df.index,
                    "Name": list(case0_rp_df.columns),
                },
            )

            for case_id in self.case_ids:
                case_iv_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'InputVars')), header=0)
                case_iv_df = case_iv_df.set_index(pd.to_datetime(case_iv_df['Date'])).drop('Date', axis=1)
                iv_xr.loc[case_id, case_iv_df.index, case_iv_df.columns] = case_iv_df.values

                case_md_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'MeaData')), header=0)
                case_md_df = case_md_df.set_index(pd.to_datetime(case_md_df['Date'])).drop('Date', axis=1)
                try:
                    md_xr.loc[case_md_df.index, case_md_df.columns] = case_md_df.values
                except Exception as e:
                    pass

                case_rsv_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'RangeStateVars')), header=0, index_col='Index')
                rsv_xr.loc[case_id, case_rsv_df.index, case_rsv_df.columns] = case_rsv_df.values

                case_ic_df = pd.read_csv(os.path.join(self.input_dir, '{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'InputCons')), header=0, index_col='Index')
                ic_xr.loc[case_id, case_ic_df.index, case_ic_df.columns] = case_ic_df.values

                rp_df = self.Data['RangeParas_ini'][case_id]
                rp_xr.loc[case_id, rp_df.index, rp_df.columns] = rp_df.values
            self.Data['InputVars'] = iv_xr
            self.Data['MeaData'] = md_xr.where(md_xr > 0)
            self.Data['RangeStateVars'] = rsv_xr
            self.Data['InputCons'] = ic_xr
            self.Data['RangeParas'] = rp_xr

            self.set_inflow_nutrient()

        # Load the repeat data (SV / Flux / Paras (RangeParas) / ParasVars)
        load_data_types = [load_data_types] if isinstance(load_data_types, str) else load_data_types
        load_data_types = [] if load_data_types is None else load_data_types
        load_repeats = {case_id: [range(self.start_repeat, self.start_repeat + self.n_repeat), range(self.start_repeat, self.start_repeat + self.n_repeat)] for case_id in self.case_ids} if load_repeats is None else load_repeats
        load_repeats = [load_repeats] if isinstance(load_repeats, int) else load_repeats
        load_repeats = dict(zip(self.case_ids, [load_repeats] * len(self.case_ids))) if isinstance(load_repeats, list) else load_repeats
        if not isinstance(load_repeats, dict):
            raise TypeError('The load_repeats must be Dic or List or None.')


        # read SV in n_repeat times and save as xarray
        if 'StateVars' in load_data_types:
            if 'StateVars' in self.Data.keys():
                # Load the repeat data according to the load_repeats
                phar = None if not if_show_bar else tqdm(total=len(load_repeats), ncols=150, desc='Load SV data')
                for case_id in load_repeats.keys():
                    for load_from_repeat_idx, load_to_repeat_idx in zip(load_repeats[case_id][0], load_repeats[case_id][1]):
                        sv_path = os.path.join(load_dir, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'StateVars', load_from_repeat_idx))
                        if os.path.exists(sv_path):
                            sv_df_i = pd.read_csv(sv_path, header=0)
                            sv_df_i = sv_df_i.set_index(pd.to_datetime(sv_df_i['Date'])).drop('Date', axis=1)
                            date_index = sv_df_i.index.intersection(self.date_range_dic[case_id])
                            self.Data['StateVars'].loc[case_id, load_to_repeat_idx, date_index, sv_df_i.columns] = sv_df_i.loc[date_index, sv_df_i.columns].values
                    _ = None if not if_show_bar else phar.update()
                _ = None if not if_show_bar else phar.close()
            else:
                # Load the ini data according to the input_dir data
                sv0_df = self.Data['StateVars_ini'][self.case_ids[0]]
                sv_data_xr = xr.DataArray(
                    np.zeros((len(self.case_ids), self.n_repeat, len(self.union_date_index), sv0_df.shape[1]), dtype=np.float32),
                    dims=("CaseID", "Repeat", "Date", "Name"),
                    coords={
                        "CaseID": self.case_ids,
                        "Repeat": range(self.start_repeat, self.start_repeat + self.n_repeat),
                        "Date": self.union_date_index,
                        "Name": list(sv0_df.columns),
                    },
                )
                for case_id in self.case_ids:
                    sv_df = self.Data['StateVars_ini'][case_id]
                    for repeat_idx in range(self.start_repeat, self.start_repeat + self.n_repeat):
                        sv_data_xr.loc[case_id, repeat_idx, sv_df.index, sv_df.columns] = sv_df.values
                self.Data['StateVars'] = sv_data_xr


        # read flux in n_repeat times and save as xarray
        if 'Flux' in load_data_types:
            if 'Flux' in self.Data.keys():
                # Load the repeat data according to the load_repeats
                phar = None if not if_show_bar else tqdm(total=len(load_repeats), ncols=150, desc='Load Flux data')
                for case_id in load_repeats.keys():
                    for load_from_repeat_idx, load_to_repeat_idx in zip(load_repeats[case_id][0], load_repeats[case_id][1]):
                        flux_path = os.path.join(load_dir, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'Flux', load_from_repeat_idx))
                        if os.path.exists(flux_path):
                            flux_df_i = pd.read_csv(flux_path, header=0)
                            flux_df_i = flux_df_i.set_index(pd.to_datetime(flux_df_i['Date'])).drop('Date', axis=1)
                            date_index = flux_df_i.index.intersection(self.date_range_dic[case_id])
                            self.Data['Flux'].loc[case_id, load_to_repeat_idx, date_index, flux_df_i.columns] = flux_df_i.loc[date_index, flux_df_i.columns].values
                    _ = None if not if_show_bar else phar.update()
                _ = None if not if_show_bar else phar.close()
            else:
                # Load the ini data according to the input_dir data
                flux_data_xr = xr.DataArray(
                    np.zeros((len(self.case_ids), self.n_repeat, len(self.union_date_index), len(flux_columns)), dtype=np.float32),
                    dims=("CaseID", "Repeat", "Date", "Name"),
                    coords={
                        "CaseID": self.case_ids,
                        "Repeat": range(self.start_repeat, self.start_repeat + self.n_repeat),
                        "Date": self.union_date_index,
                        "Name": flux_columns,
                    },
                )
                self.Data['Flux'] = flux_data_xr


        # Load paras with n_repeat and save as df
        if 'Paras' in load_data_types:
            ratio_names = ['Ratio_Chla2C_{}'.format(f) for f in F_types]

            if 'Paras' in self.Data.keys():
                # Load the repeat data according to the load_repeats
                phar = None if not if_show_bar else tqdm(total=len(load_repeats), ncols=150, desc='Load Paras data')
                for case_id in load_repeats.keys():
                    for load_from_repeat_idx, load_to_repeat_idx in zip(load_repeats[case_id][0], load_repeats[case_id][1]):
                        paras_path = os.path.join(load_dir, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], 'Paras', load_from_repeat_idx))
                        if os.path.exists(paras_path):
                            paras_df_i = pd.read_csv(paras_path, header=0, index_col='Index')
                            self.Data['Paras'].loc[case_id, load_to_repeat_idx, paras_df_i.index] = paras_df_i['Value'].values
                            for ratio_name in ratio_names:
                                try:
                                    self.Data['ChlaRatio'].loc[case_id, :, :, ratio_name] = paras_df_i.loc[ratio_name, 'Value']
                                except Exception as e:
                                    self.Data['ChlaRatio'].loc[case_id, :, :, ratio_name] = 0
                    _ = None if not if_show_bar else phar.update()
                _ = None if not if_show_bar else phar.close()
            else:
                # Load the ini paras values according to the input_dir data
                paras0_df = self.Data['Paras_ini'][self.case_ids[0]]
                paras_data_xr = xr.DataArray(
                    np.full((len(self.case_ids), self.n_repeat, paras0_df.shape[0]), np.nan, dtype=np.float32),
                    dims=("CaseID", "Repeat", "Name"),
                    coords={
                        "CaseID": self.case_ids,
                        "Repeat": range(self.start_repeat, self.start_repeat + self.n_repeat),
                        "Name": list(paras0_df.index),
                    },
                )

                ratio_xr = xr.DataArray(
                    np.full((len(self.case_ids), self.n_repeat, len(self.union_date_index), len(ratio_names)), np.nan, dtype=np.float32),
                    dims=("CaseID", "Repeat", "Date", "Name"),
                    coords={
                        "CaseID": self.case_ids,
                        "Repeat": range(self.start_repeat, self.start_repeat + self.n_repeat),
                        "Date": self.union_date_index,
                        "Name": ratio_names,
                    },
                )

                for case_id in self.case_ids:
                    case_paras_df = self.Data['Paras_ini'][case_id]
                    for repeat_idx in range(self.start_repeat, self.start_repeat + self.n_repeat):
                        paras_data_xr.loc[case_id, repeat_idx, case_paras_df.index] = case_paras_df['Value'].values
                    for ratio_name in ratio_names:
                        try:
                            ratio_xr.loc[case_id, :, :, ratio_name] = case_paras_df.loc[ratio_name, 'Value']
                        except Exception as e:
                            print(e)

                self.Data['Paras'] = paras_data_xr
                self.Data['ChlaRatio'] = ratio_xr


        return self.Data

    def load_modules(self):
        limitations = {
            'fn': NitrogenLimitation(Model=self),
            'fp': PhosphorusLimitation(Model=self),
            'fl': LightLimitation(Model=self),
            'ft': TemperatureLimitation(Model=self),
            'fo': OxygenLimitation(Model=self),
        }

        # load modules
        self.WaterBalance = LakeWaterBalance(
            Model=self,
            components=['WL'],
            based_element='H2O',
            flux_ins_dic=FluxInDic,
            flux_outs_dic=FluxOutDic,
            _class='TW'
        )
        self.Phytoplankton = LakePhytoplankton(
            Model=self,
            components=F_types,
            based_element='C',
            flux_ins_dic=FluxInDic,
            flux_outs_dic=FluxOutDic,
            _class='TF',
            limitations_dic=limitations,
            process_limitations_dic={
                'Grow': {'ft': 0, 'fn': 0, 'fp': 0, 'fl': 0},
                'Mortality': {'ft': 1},
                'Excretion': {'ft': 1},
                'Predation': {'ft': 1},
            },
            derived_elements='Chla',
        )

        self.DissolvedOxygen = LakeDissolvedOxygen(
            Model=self,
            components=['DO'],
            based_element='O',
            flux_ins_dic=FluxInDic,
            flux_outs_dic=FluxOutDic,
            _class='TO',
            linked_biology_ls=[self.Phytoplankton]
        )

        self.Nitrogen = LakeNitrogen(
            Model=self,
            components=N_types,
            based_element='N',
            flux_ins_dic=FluxInDic,
            flux_outs_dic=FluxOutDic,
            _class='TN',
            limitations_dic=limitations,
            process_limitations_dic={
                'Mineralization': {'ft': 1},
                'Nitrification': {'ft': 0, 'fn': 0, 'fo': 0},
                'Denitrification': {'ft': 0, 'fn': 0, 'fo': 0},
                'Releasing': {'ft': 1},
                'Resuspension': {'ft': 1},
            },
            linked_biology_metabolism_dic={self.Phytoplankton: ['Mortality', 'Excretion', 'Predation']},
            linked_biology_uptake_dic={self.Phytoplankton: ['Grow']}
        )

        self.Phosphorus = LakePhosphorus(
            Model=self,
            components=P_types,
            based_element='P',
            flux_ins_dic=FluxInDic,
            flux_outs_dic=FluxOutDic,
            _class='TP',
            limitations_dic=limitations,
            process_limitations_dic={
                'Mineralization': {'ft': 1},
                'Releasing': {'ft': 1},
                'Resuspension': {'ft': 1},
            },
            linked_biology_metabolism_dic={self.Phytoplankton: ['Mortality', 'Excretion', 'Predation']},
            linked_biology_uptake_dic={self.Phytoplankton: ['Grow']}
        )

    def save_data(self, save_dir, save_cases=None, save_repeats=None, data_names=None, if_show_bar=False):
        """
        Purpose: Save [data_names] in [save_cases]'s [save_repeats] to CSV at Dir [save_dir]
            Saved data names: 'Flux', 'StateVars', 'ParasVars', Paras
        """
        save_repeats = range(self.start_repeat, self.start_repeat + self.n_repeat) if save_repeats is None else save_repeats
        save_repeats = [save_repeats] if isinstance(save_repeats, int) else save_repeats
        save_cases = [save_cases] if isinstance(save_cases, str) else save_cases
        save_cases = self.case_ids if save_cases is None else save_cases
        data_names = ['Flux', 'StateVars', 'Paras'] if data_names is None else data_names
        data_names = [data_names] if isinstance(data_names, str) else data_names

        _ = None if os.path.exists(save_dir) else os.mkdir(save_dir)
        pbar = tqdm(ncols=150, total=len(save_cases), desc='Save {} cases"s {} to CSV'.format(len(save_cases), '-'.join(data_names))) if if_show_bar else None
        for case_id in save_cases:
            for data_name in data_names:
                data = self.Data[data_name]
                if data_name in ['Flux', 'StateVars', 'ParasVars', 'InputVars_Repeat']:
                    for repeat_idx in save_repeats:
                        df = data.loc[case_id, repeat_idx, self.date_range_dic[case_id], :].to_pandas().loc[self.union_date_index, :]
                        df.to_csv(os.path.join(save_dir, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], data_name, repeat_idx)))
                elif data_name == 'Paras':
                    for repeat_idx in save_repeats:
                        df = pd.DataFrame({'Index': list(data.coords['Name'].values), 'Value': data.loc[case_id, repeat_idx, :].values}).set_index('Index')
                        df.to_csv(os.path.join(save_dir, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], data_name, repeat_idx)))
                else:
                    df = data.loc[case_id, self.date_range_dic[case_id], :].to_pandas().loc[self.union_date_index, :]
                    df.to_csv(os.path.join(save_dir, '{}-{}-{}-{}.csv'.format(case_id, AllCases_dic[case_id], data_name, '0')))
            _ = pbar.update() if if_show_bar else None
        _ = pbar.close() if if_show_bar else None

    def set_data(self, data_dic):
        """
        Set data for the model
        data_dic: Dict
            {case_id: {data_name: [[repeat_idx, ], [data_df, ]], }, }
        """
        data_dic = {} if not isinstance(data_dic, dict) else data_dic
        for case_id, case_data_dic in data_dic.items():
            for case_data_name, case_data_ls in case_data_dic.items():
                if case_data_name in ['Flux', 'StateVars', 'ParasVars']:
                    for set_to_repeat_idx, set_to_repeat_df in zip(case_data_ls[0], case_data_ls[1]):
                        self.Data[case_data_name].loc[case_id, set_to_repeat_idx, set_to_repeat_df.index, set_to_repeat_df.columns] = set_to_repeat_df.values
                elif case_data_name == 'Paras':
                    for set_to_repeat_idx, set_to_repeat_df in zip(case_data_ls[0], case_data_ls[1]):
                        self.Data[case_data_name].loc[case_id, set_to_repeat_idx, set_to_repeat_df.index] = set_to_repeat_df['Value'].values
                else:
                    pass

    def set_inflow_nutrient(self):
        lake_attr_df = pd.read_csv('Data/Lake mean attributes.csv', index_col=0, header=0)
        for case_id in self.case_ids:
            depth, retention = lake_attr_df.loc[case_id, 'Depth'], lake_attr_df.loc[case_id, 'Retention']
            retention /= 365.25
            mea_tn_df = pd.read_csv(os.path.join('Data/MeaData', '{}-TN-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_tn_df.index = pd.to_datetime(mea_tn_df.index)
            mea_tp_df = pd.read_csv(os.path.join('Data/MeaData', '{}-TP-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_tp_df.index = pd.to_datetime(mea_tp_df.index)

            for year in mea_tn_df.index.year.unique():
                year_mean_lake_tn = mea_tn_df[mea_tn_df.index.year == year]['TN'].mean()
                year_mean_inflow_tn = year_mean_lake_tn / (0.34 * np.power(retention, -0.16) * np.power(depth, 0.17))
                for n_type, percent in Initial_distribute_dics['TN'].items():
                    self.Data['InputVars'].loc[case_id, '{}-1-1'.format(year): '{}-12-31'.format(year), '{}_Inflow'.format(n_type)] = year_mean_inflow_tn * percent

            for year in mea_tp_df.index.year.unique():
                year_mean_lake_tp = mea_tp_df[mea_tp_df.index.year == year]['TP'].mean()
                year_mean_inflow_tp = year_mean_lake_tp * (1 + np.power(retention, 0.5))
                for p_type, percent in Initial_distribute_dics['TP'].items():
                    self.Data['InputVars'].loc[case_id, '{}-1-1'.format(year): '{}-12-31'.format(year), '{}_Inflow'.format(p_type)] = year_mean_inflow_tp * percent

    def set_start_repeat(self, new_start_repeat):
        """
        Set the start repeat index for repeatable data.
        """
        self.start_repeat = new_start_repeat
        self.repeat_range = range(self.start_repeat, self.start_repeat + self.n_repeat)
        for data_name in ['StateVars', 'Flux', 'Paras', 'ParasVars']:
            if data_name in self.Data.keys():
                self.Data[data_name].coords['Repeat'] = self.repeat_range


if __name__ == '__main__':
    """
    Test: Run model with ini paras
    """
    t = Model(
        input_dir='Data/ModelInput',
        case_ids=list(UsedCases_dic.keys())[:5],
        n_repeat=1,
        start_repeat=0,
        deal_range_out='clip'
    )
    t.run(if_show_bar=True)
    t.save_data(
        save_dir='Results/ModelSimulationResults',
        data_names=['Flux', 'StateVars', 'Paras'],
        if_show_bar=True
    )
