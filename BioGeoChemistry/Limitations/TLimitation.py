import math
from Config import *

import numpy as np


class TemperatureLimitation:
    def __init__(self, Model=None):
        self.Model = Model

    def type_0(self, component, process_name, theta=-0.01, theta2=None, **kwargs):
        """
        Refer EFDC.
        Shape like:
            np.exp(theta1 * np.power(T_Ave - T_Opt, 2))     if T_Ave < T_Opt
            1                                               if T_Ave == Opt_T
            np.exp(theta2 * np.power(T_Ave - T_Opt, 2))     if T_Ave > T_Opt
        """
        theta2 = theta if theta2 is None else theta2
        T_Opt = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Opt_T_{}_{}'.format(process_name, component)].values
        T_Ave = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Ave'].values.reshape(-1, 1)
        return np.where(T_Ave < T_Opt, np.exp(theta * np.power(T_Ave - T_Opt, 2)), np.exp(theta2 * np.power(T_Ave - T_Opt, 2)))

    def type_1(self, component, process_name, theta=0.05, **kwargs):
        """
        Refer EFDC.
        Shape like: S
            np.exp(theta * (T_Ave - Opt_T))
        """
        T_Ave = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Ave'].values.reshape(-1, 1)
        try:
            T_Opt = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Opt_T_{}_{}'.format(process_name, component)].values
        except Exception as e:
            T_Opt = 20
        return np.exp(theta * (T_Ave - T_Opt))

    def type_2(self, component, process_name, theta=-4.6, theta2=None, **kwargs):
        """
        Refer Svr. 1984
        Shape like:
            Q1 = TOpt - TMin
            Q2 = TMax - TOpt
            math.exp(theta * math.pow((TOpt-TAve)/Q1, 4))   if TAve < TOpt
            math.exp(theta2 * math.pow((TAve-TOpt)/Q2, 4))   if TAve > TOpt
        """
        theta2 = theta if theta2 is None else theta2
        T_Opt = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Opt_T_{}_{}'.format(process_name, component)].values
        if self.Model.use_repeat_IV:
            T_Ave = self.Model.Data['InputVars_Repeat'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'T_Ave'].values
            T_Min = self.Model.Data['InputVars_Repeat'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'T_Min'].values
            T_Max = self.Model.Data['InputVars_Repeat'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'T_Max'].values
        else:
            T_Ave = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Ave'].values.reshape(-1, 1)
            T_Min = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Min'].values.reshape(-1, 1)
            T_Max = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Max'].values.reshape(-1, 1)
        Q1 = T_Opt - T_Min
        Q2 = T_Max - T_Opt
        return np.where(T_Ave < T_Opt, np.exp(theta * np.power((T_Opt - T_Ave) / Q1, 4)), np.exp(theta2 * np.power((T_Ave - T_Opt) / Q2, 4)))

    def Denitrification(self, component, type_idx=0, **kwargs):
        """
        theta:
            Self: -0.005
        Opt_T:
            Self: 30
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = -0.005 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Mineralization(self, component, type_idx=1, **kwargs):
        """
        theta:
            EFDC: 0.07
        Opt_T: reference temperature
            EFDC: 20
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = 0.07 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Resuspension(self, component, type_idx=1, **kwargs):
        """
        theta:
            Self: 0.05
        Opt_T: reference temperature
            Self: 20
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = 0.05 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Releasing(self, component, type_idx=1, **kwargs):
        """
        theta:
            Self: 0.05
        Opt_T: reference temperature
            Self: 20
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = 0.05 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Nitrification(self, component, type_idx=0, **kwargs):
        """
        theta:
            EFDC: -0.003
        Opt_T:
            EFDC: 30
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = -0.003 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Grow(self, component, type_idx=0, **kwargs):
        """
        theta, theta2:
            EFDC: -0.004, -0.012
        Opt_T:
            EFDC: 25
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = -0.004 if 'theta' not in kwargs.keys() else kwargs['theta']
        theta2 = -0.012 if 'theta2' not in kwargs.keys() else kwargs['theta2']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta, theta2=theta2)

    def Mortality(self, component, type_idx=1, **kwargs):
        """
        theta:
            EFDC: 0.032
        Opt_T:
            EFDC: 20
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = 0.032 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Predation(self, component, type_idx=1, **kwargs):
        """
        theta:
            EFDC: 0.032
        Opt_T:
            EFDC: 20
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = 0.032 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)

    def Excretion(self, component, type_idx=1, **kwargs):
        """
        theta:
            EFDC: 0.032
        Opt_T:
            EFDC: 20
        """
        process_name = inspect.currentframe().f_code.co_name
        theta = 0.032 if 'theta' not in kwargs.keys() else kwargs['theta']
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, theta=theta)
