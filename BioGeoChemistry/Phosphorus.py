import numpy as np
from BioGeoChemistry.base import Element
from BioGeoChemistry.Limitations.TLimitation import TemperatureLimitation
from Config import *


"""
P BGC(mine): ft
    0: Pond-NP ()
    1: Huang 2018 / Cui 2022 (_H)
    2: EFDC (_EF)
    3: PCLake (_PL)
"""


# phosphorus biogeochemical process
class LakePhosphorus(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def Mineralization(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'OP':
            self.update_type_0(process_name=process_name, component=component)

    def Settling(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'OP':
            self.update_type_0(process_name=process_name, component=component, ratio=1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)

    def Releasing(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'PO4':
            self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)

    def Resuspension(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'OP':
            self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)

    def Adsorption(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'PO4':
            self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)


if __name__ == '__main__':
    t = LakePhosphorus
