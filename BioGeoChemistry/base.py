from Config import *


# describe the matter using concentration
class Matter:
    """
    The base class of Concentration Matter in Water.
    Attr:
        Model: model obj
            linked model.
        _class: str
            the matter class. like diatom's class is phytoplankton, NH3 class is nitrogen.
        components: list
            the matter's components. like nitrogen contains ON, NOx, NH3.
        based_element: str
            the matter weight value based element. like phytoplankton based on Carbon.
        flux_ins_dic: dict
            every component's flux in processes. it is a dic like {component1: ['{}_{}'.format(process1, component1)], component2: ['{}_{}'.format(process2, component2)]}.
        flux_outs_dic: dict
            every component's flux out processes. shape like flux in.
        limitations_dic: dict
            the limitations of the processes in every component. it is a dic like {limitation1_name: limitation1_Obj}
        process_limitations_dic: dict
            the considered limitations and its use type in processes of a component. it is a dic like {process1: {limitation1_name: use_type_idx}}
    Function:
        dynamics: update the flux, return None
            calc the matter dynamics in water during one step.
        mix: update the flux, return None
            calc the matter dynamics due to the water column changes.
        aggregate: update the flux, return None
            calc the components sum in the matter. like calc TP = PO4 + OP.
    """

    def __init__(self, **kwargs):
        self.Model = kwargs['Model'] if 'Model' in kwargs.keys() else None
        self._class = kwargs['_class'] if '_class' in kwargs.keys() else None
        self.components = kwargs['components'] if 'components' in kwargs.keys() else None
        self.based_element = kwargs['based_element'] if 'based_element' in kwargs.keys() else None
        self.flux_ins_dic = kwargs['flux_ins_dic'] if 'flux_ins_dic' in kwargs.keys() else None
        self.flux_outs_dic = kwargs['flux_outs_dic'] if 'flux_outs_dic' in kwargs.keys() else None
        self.limitations_dic = kwargs['limitations_dic'] if 'limitations_dic' in kwargs.keys() else None
        self.process_limitations_dic = kwargs['process_limitations_dic'] if 'process_limitations_dic' in kwargs.keys() else None

    def update_type_0(self, component, process_name, ratio=1):
        """
        Update the flux with a process function like:
            Flux = rate * lim[0] * lim[1] ... * ratio
        """
        rate = self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Rate_{}_{}'.format(process_name, component)].values * ratio
        if process_name in self.process_limitations_dic.keys():
            for limitation, type_idx in self.process_limitations_dic[process_name].items():
                rate *= getattr(self.limitations_dic[limitation], process_name)(component, type_idx=type_idx)
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = rate * self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values

    def dynamics(self):
        for component in self.components:
            for flux_in in self.flux_ins_dic[component]:
                getattr(self, '{}'.format(flux_in[: flux_in.index('_')]))(component)
            for flux_out in self.flux_outs_dic[component]:
                getattr(self, '{}'.format(flux_out[: flux_out.index('_')]))(component)

    def Mix(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        """
        The conc. changes in water level dynamics
        (X_inflow * Flux[inflow] + X * StateVars[wl, last date] + X_pr * Flux[pr]) / StateVars[wl, crt date]
        """
        component_last_date = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values * self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values
        inflow = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, '{}_Inflow'.format(component)].values.reshape(-1, 1)
        pr = self.Model.Data['InputVars'].loc[self.Model.crt_date_run_cases, self.Model.crt_date, '{}_Pr'.format(component)].values.reshape(-1, 1)

        component_inflow = inflow * self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'Inflow_WL'].values
        component_pr = pr * self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'Precipitation_WL'].values
        wl_last_date = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, 'WL'].values
        wl_inflow = self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'Inflow_WL'].values
        wl_pr = self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'Precipitation_WL'].values

        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = (component_last_date + component_inflow + component_pr) / (wl_last_date + wl_inflow + wl_pr) - self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values

    def aggregate(self):
        for component in self.components:
            self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'From{}_{}'.format(component, self._class)] = self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, component].values - self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values


class Biology(Matter):
    """
    The concentration biology in water.
    Attr:
        derived_elements: list or str
            the element need derived base on the biology. like calc the chl a from phytoplankton components
    Function:
        derive: update the flux, return None
            calc the derived value of the derived_element in every component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'derived_elements' in kwargs.keys():
            if isinstance(kwargs['derived_elements'], list):
                self.derived_elements = kwargs['derived_elements']
            elif isinstance(kwargs['derived_elements'], str):
                self.derived_elements = [kwargs['derived_elements']]
            else:
                raise ValueError('Invalid paras value of derived_elements, it should be list or int or None')
        else:
            self.derived_elements = []

    def derive(self):
        for derived_element in self.derived_elements:
            for component in self.components:
                self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, 'In{}_{}'.format(component, derived_element)] = (self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, component].values - self.Model.Data['StateVars'].loc[self.Model.crt_date_run_cases, :, self.Model.last_date, component].values) * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Ratio_{}2{}_{}'.format(derived_element, self.based_element, component)].values


class Element(Matter):
    """
    The concentrated element in Water. like N and P
    Attr:
        linked_biology_metabolism_dic: dict
            the biology metabolism processes that well release as the component in the element. shape like {biology_obj: [biology_metabolism_process1, biology_metabolism_process2]}.
        linked_biology_uptake_dic: dict
            the biology processes that well uptake the component in the element. shape like {biology_name: [biology_uptake_process1]}.
    Function:
        biology_metabolism: update the flux, return None
            calc the flux of biology metabolism released component in the element.
        biology_uptake: update the flux, return None
            calc the flux of biology uptake consumed component in the element.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'linked_biology_metabolism_dic' in kwargs.keys():
            self.linked_biology_metabolism_dic = kwargs['linked_biology_metabolism_dic'] if isinstance(kwargs['linked_biology_metabolism_dic'], dict) else {}
        else:
            self.linked_biology_metabolism_dic = {}

        if 'linked_biology_uptake_dic' in kwargs.keys():
            self.linked_biology_uptake_dic = kwargs['linked_biology_uptake_dic'] if isinstance(kwargs['linked_biology_uptake_dic'], dict) else {}
        else:
            self.linked_biology_uptake_dic = {}

    def BiologyMetabolism(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        total = 0
        for biology_obj, processes in self.linked_biology_metabolism_dic.items():
            for bio_component in biology_obj.components:
                for process in processes:
                    total += self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process, bio_component)].values * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Ratio_{}2{}_{}'.format(self.based_element, biology_obj.based_element, bio_component)].values * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Ratio_{}2{}_{}'.format(component, self.based_element, bio_component)].values
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = total

    def BiologyUptake(self, component, **kwargs):
        process_name = inspect.currentframe().f_code.co_name
        total = 0
        for biology_obj, processes in self.linked_biology_uptake_dic.items():
            for bio_component in biology_obj.components:
                for process in processes:
                    total += self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process, bio_component)].values * self.Model.Data['Paras'].loc[self.Model.crt_date_run_cases, :, 'Ratio_{}2{}_{}'.format(self.based_element, biology_obj.based_element, bio_component)].values
        self.Model.Data['Flux'].loc[self.Model.crt_date_run_cases, :, self.Model.crt_date, '{}_{}'.format(process_name, component)] = total
