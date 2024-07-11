from Config import *


class PhosphorusLimitation:
    def __init__(self, Model):
        self.Model = Model

    def type_0(self, component, process_name, p, **kwargs):
        """
        Refer EFDC.
        Shape like:
            p / (p + hsc)
        """
        hsc = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'HSC_P_{}_{}'.format(process_name, component)].values
        return p / (hsc + p)

    def Grow(self, component, type_idx=0, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        p = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'PO4'].values
        return getattr(self, 'type_{}'.format(type_idx))(component, process_name, p)
