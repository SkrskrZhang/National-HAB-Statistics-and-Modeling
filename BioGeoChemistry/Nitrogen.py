from Config import *
from BioGeoChemistry.base import Element


"""     
NP BGC(mine, nit, denit): ft, fn, fp, fdo
    0: Pond-NP ()
    1: Huang 2018 / Cui 2022 (_H)
    2: EFDC (_EF)
    3: PCLake (_PL)
"""


# lake nitrogen biogeochemical process
class LakeNitrogen(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def Mineralization(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'ON':
            self.update_type_0(process_name=process_name, component=component)

    def Nitrification(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'NH3':
            self.update_type_0(process_name=process_name, component=component)

    def Denitrification(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'NOx':
            self.update_type_0(process_name=process_name, component=component, ratio=1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)

    def Anammox(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'NH3':
            self.update_type_0(process_name=process_name, component=component, ratio=1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)

    def Deposition(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)

    def Settling(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'ON':
            self.update_type_0(process_name=process_name, component=component, ratio=1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)

    def Releasing(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component in ['NOx', 'NH3']:
            self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)

    def Resuspension(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component == 'ON':
            self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)

    def Adsorption(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        if component in ['NOx', 'NH3']:
            self.update_type_0(process_name=process_name, component=component, ratio=(1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)/self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)

    def BiologyUptake(self, component, **kwargs):
        """
        Consider the preference of F uptake NH3 than NOx, refer EFDC.
        """
        NH3 = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'NH3'].values
        NOx = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'NOx'].values
        process_name = inspect.currentframe().f_code.co_name
        total = 0
        for biology_obj, processes in self.linked_biology_uptake_dic.items():
            for bio_component in biology_obj.components:
                for process in processes:
                        HSC_N = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'HSC_N_{}_{}'.format(process, bio_component)].values
                        PN = NH3 * (NOx / ((HSC_N + NH3) * (HSC_N + NOx))) + NH3 * (HSC_N / ((NH3 + NOx) * (HSC_N + NOx)))
                        if component == 'NH3':
                            PN = PN
                        elif component == 'NOx':
                            PN = 1 - PN
                        total += self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process, bio_component)].values * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Ratio_{}2{}_{}'.format(self.based_element, biology_obj.based_element, bio_component)].values * PN
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = total


if __name__ == '__main__':
    t = LakeNitrogen

