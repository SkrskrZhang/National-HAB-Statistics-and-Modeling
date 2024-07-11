import numpy as np
from Config import *


class LightLimitation:
    def __init__(self, Model):
        self.Model = Model

    def type_0(self, component, process_name, **kwargs):
        """
        Refer to EFDC Steele’s equation.
        Shape like:
            (I / I_Opt) * exp(1 - (I / I_Opt))
        """
        I = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'SR_Sum'].values.reshape(-1, 1)
        Opt_I = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Opt_SR_{}_{}'.format(process_name, component)].values
        return (I / Opt_I) * np.exp(1. - (I / Opt_I))

    def type_1(self, component, process_name, **kwargs):
        # Calc the opti light
        I = (self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'SR_Sum'].values.reshape(-1, 1) * 1e6) / 41868.0  # convert MJ/D to Langleys/D
        I = np.tile(I, (1, self.Model.n_repeat))
        I_last = (self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.last_date, 'SR_Sum'].values.reshape(-1, 1) * 1e6) / 41868.0
        I_last = np.tile(I_last, (1, self.Model.n_repeat))
        calc_opt_I = 0.7 * I + 0.3 * I_last
        min_opt_I = (self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Opt_SR_{}_{}'.format(process_name, component)].values * 1e6) / 41868.0
        opt_I = np.where(min_opt_I < calc_opt_I, calc_opt_I, min_opt_I)

        # Light attenuation (back + chla)
        k_ext_back = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'K_EXT_Back'].values
        k_ext_chla = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'K_EXT_Chla'].values
        k_ext_f = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'K_EXT_{}'.format(component)].values
        chla = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'Chla'].values * 1000.0  # to ug/L
        k_ext_segment = k_ext_back + k_ext_chla * chla
        k_ext_segment = np.where(k_ext_segment > 0.9, 0.9, k_ext_segment)
        k_ext_f = k_ext_segment * k_ext_f

        # the optimal light for phytoplankton growth under adaptable water depth
        opt_light = opt_I * np.exp(-k_ext_f * 1.0)

        # calc day in year
        day_in_year = self.Model.crt_date.day_of_year
        lat = np.tile(self.Model.Data['InputCons'].loc[self.Model.crt_date_run_cases, 'Lat', 'Value'].values.reshape(-1, 1), (1, self.Model.n_repeat))
        delta = 23.45 * np.sin(np.deg2rad(360. * ((284. + day_in_year) / 365.)))
        omega = np.degrees(np.arccos(-1. * np.tan(np.deg2rad(lat)) * np.tan(np.deg2rad(delta))))
        day_length = (2.0 * omega) / 15.
        fractional_day_length = day_length / 24.0

        # a/b in light function
        dz = 0.0
        depth = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values
        depth = np.where(depth > Max_depth_for_light_limitation, Max_depth_for_light_limitation, depth)
        a = -(I / (fractional_day_length * opt_light)) * np.exp(-k_ext_f * (dz + depth))
        b = -(I / (fractional_day_length * opt_light)) * np.exp(-k_ext_f * dz)

        # light function results
        light_factor = ((2.718 * fractional_day_length) / (k_ext_f * depth)) * (np.exp(a) - np.exp(b))
        return light_factor

    def Grow(self, component, type_idx=0, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name)