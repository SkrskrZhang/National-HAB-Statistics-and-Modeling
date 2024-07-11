from Config import *
from BioGeoChemistry.base import Element


# crab pond dissolved oxygen dynamic
class LakeDissolvedOxygen(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linked_biology_ls = kwargs['linked_biology_ls'] if 'linked_biology_ls' in kwargs.keys() else []

    def Deposition(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        T_Ave = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Ave'].values.reshape(-1, 1)

        DO_Sat = 14.61996 - (0.4042 * T_Ave) + (0.00842 * np.power(T_Ave, 2)) - (0.00009 * np.power(T_Ave, 3))
        rate = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Rate_{}_{}'.format(process_name, component)].values * (DO_Sat - self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values)
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = rate

    def get_SaturatedDO(self):
        T_Ave = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, 'T_Ave'].values.reshape(-1, 1)

        return 14.61996 - (0.4042 * T_Ave) + (0.00842 * np.power(T_Ave, 2)) - (0.00009 * np.power(T_Ave, 3))

    def BiologyRespiration(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        total = 0
        for biology in self.linked_biology_ls:
            for bio_component in biology.components:
                total += self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, bio_component].values * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Rate_Respiration_{}'.format(bio_component)].values
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = total

    def BiologyPhotosynthesis(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        total = 0
        for biology in self.linked_biology_ls:
            for bio_component in biology.components:
                total += self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, bio_component].values * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Rate_Photosynthesis_{}'.format(bio_component)].values
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = total
