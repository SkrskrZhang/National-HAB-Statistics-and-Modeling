from Config import *


class OxygenLimitation:
    def __init__(self, Model):
        self.Model = Model

    def type_0(self, component, process_name, DO=None, **kwargs):
        """
        For nitrification
        Refer EFDC.
        Shape like:
            DO / (DO + hsc)
        """
        DO = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'DO'].values if DO is None else DO
        hsc = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'HSC_O_{}_{}'.format(process_name, component)].values
        return DO / (hsc + DO)

    def type_2(self, component, process_name, DO=None, **kwargs):
        """
        For denitrification
        Refer EFDC.
        Shape like:
            hsc / (DO + hsc)
        """
        DO = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'DO'].values if DO is None else DO
        hsc = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'HSC_O_{}_{}'.format(process_name, component)].values
        return hsc / (hsc + DO)

    def Denitrification(self, component, type_idx=1, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name)

    def Nitrification(self, component, type_idx=0, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name)
