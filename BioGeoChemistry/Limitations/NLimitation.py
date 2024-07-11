from Config import *


class NitrogenLimitation:
    def __init__(self, Model):
        self.Model = Model

    def type_0(self, component, process_name, n, **kwargs):
        """
        Refer EFDC.
        Shape like:
            n / (n + hsc)
        """
        hsc = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'HSC_N_{}_{}'.format(process_name, component)].values
        return n / (hsc + n)

    def Grow(self, component, type_idx=0, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        n = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'NH3'].values + self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'NOx'].values
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, n)

    def Denitrification(self, component, type_idx=0, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        n = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'NOx'].values
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, n)

    def Nitrification(self, component, type_idx=0, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        n = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'NH3'].values
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, n)

