from Config import *
from BioGeoChemistry.base import Biology


class LakePhytoplankton(Biology):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def Grow(self, component, **kwargs):
        """
        Refer to Huang. 2012
        dt = Rate * fi * ft * min(fn, fp)
        """
        process_name = inspect.currentframe().f_code.co_name
        fn = self.limitations_dic['fn'].Grow(component)
        fp = self.limitations_dic['fp'].Grow(component)
        self.update_type_0(process_name=process_name, component=component, ratio=1. / np.where(fn > fp, fn, fp))

    def Mortality(self, component, **kwargs):
        """
        Refer to EFDC
        dt = Rate * ft
        """
        process_name = inspect.currentframe().f_code.co_name
        self.update_type_0(process_name=process_name, component=component)

    def Excretion(self, component, **kwargs):
        """
        Refer to EFDC
        dt = Rate * ft
        """
        process_name = inspect.currentframe().f_code.co_name
        self.update_type_0(process_name=process_name, component=component)

    def Predation(self, component, **kwargs):
        """
        Refer to EFDC
        dt = Rate * ft * fa (fa: the available F for Z eat)
        """
        process_name = inspect.currentframe().f_code.co_name
        ratio = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values / self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'HSC_C_Predation_{}'.format(component)].values
        self.update_type_0(process_name=process_name, component=component, ratio=ratio)

    def Settling(self, component, **kwargs):
        """
        Refer to EFDC
        dt = Rate / WL
        """
        process_name = inspect.currentframe().f_code.co_name
        self.update_type_0(process_name=process_name, component=component, ratio=1. / self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values)

    def derive(self):
        """
        F derive Chla, while the Ratio_Chla2F is changeable, it is that the most reliable ratio was used compared to the measured Chla
        :return:
        """
        for derived_element in self.derived_elements:
            Fs = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, self.components].values
            ratios = self.Model.Data['{}Ratio'.format(derived_element)].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, ['Ratio_{}2{}_{}'.format(derived_element, self.based_element, component) for component in self.components]].values
            self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, derived_element] = (Fs * ratios).sum(axis=-1)





