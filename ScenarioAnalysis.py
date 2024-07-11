from Config import *
from NationalHABModel import Model
from ModelOptimization import load_calibrated_paras


class ScenarioModelOfWithoutNutrientReductionDuring2005to2022:
    def __init__(self, dir_calibrated_paras):
        model = Model(
            input_dir='Data/ModelInput',
            case_ids=list(UsedCases_dic.keys()),
            n_repeat=1,
            start_repeat=0,
            deal_range_out='clip'
        )
        self.calibrated_model = load_calibrated_paras(model, dir_calibrated_paras)

    def set_inflow_nutrient_without_reduction_during_2005_2022(self):
        for case_id in self.calibrated_model.case_ids:
            mea_tn_df = pd.read_csv(os.path.join('Data/MeaData', '{}-TN-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_tn_df.index = pd.to_datetime(mea_tn_df.index)
            mea_tp_df = pd.read_csv(os.path.join('Data/MeaData', '{}-TP-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_tp_df.index = pd.to_datetime(mea_tp_df.index)

            # First year inflow nutrient
            first_year_mean_tn, first_year_mean_tp = mea_tn_df['TN'].values[:12].mean(), mea_tn_df['TP'].values[:12].mean()

            for n_type, percent in Initial_distribute_dics['TN'].items():
                self.calibrated_model.Data['InputVars'].loc[case_id, :, '{}_Inflow'.format(n_type)] = first_year_mean_tn * percent
            for p_type, percent in Initial_distribute_dics['TP'].items():
                self.calibrated_model.Data['InputVars'].loc[case_id, :, '{}_Inflow'.format(p_type)] = first_year_mean_tp * percent

    def run(self, save_dir):
        # Set inflow nutrient
        self.set_inflow_nutrient_without_reduction_during_2005_2022()

        # Run and save
        self.calibrated_model.run(if_show_bar=True)
        self.calibrated_model.save_data(
            save_dir=save_dir,
            data_names=['StateVars'],
            if_show_bar=True
        )


class ScenarioModelOfFuture:
    def __init__(self, dir_calibrated_paras, climate_scenarios, nutrient_scenarios):
        self.model_dic = {}
        for climate_scenario in climate_scenarios:
            for nutrient_scenario in nutrient_scenarios:
                model = Model(
                    input_dir='Data/ModelInputFuture{}'.format(climate_scenario),
                    case_ids=list(UsedCases_dic.keys()),
                    n_repeat=1,
                    start_repeat=0,
                    deal_range_out='clip'
                )
                calibrated_model = load_calibrated_paras(model, dir_calibrated_paras)
                calibrated_model = self.set_inflow_nutrient(calibrated_model, nutrient_scenario)
                self.model_dic['{}-{}'.format(climate_scenario, nutrient_scenario)] = calibrated_model

    @staticmethod
    def set_inflow_nutrient(calibrated_model, nutrient_scenario):
        lake_water_quality_target_df = pd.read_csv('Data/Lake mean attributes.csv', index_col=0, header=0)

        for case_id in calibrated_model.case_ids:
            mea_tn_df = pd.read_csv(os.path.join('Data/MeaData', '{}-TN-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_tn_df.index = pd.to_datetime(mea_tn_df.index)
            mea_tp_df = pd.read_csv(os.path.join('Data/MeaData', '{}-TP-Mean.csv'.format(case_id)), index_col=0, header=0).dropna()
            mea_tp_df.index = pd.to_datetime(mea_tp_df.index)

            # Last year inflow nutrient
            last_year_mean_tn, last_year_mean_tp = mea_tn_df['TN'].values[-12:].mean(), mea_tn_df['TP'].values[-12:].mean()
            if nutrient_scenario == 'WithoutNutrientReduction':
                for n_type, percent in Initial_distribute_dics['TN'].items():
                    calibrated_model.Data['InputVars'].loc[case_id, :, '{}_Inflow'.format(n_type)] = last_year_mean_tn * percent
                for p_type, percent in Initial_distribute_dics['TP'].items():
                    calibrated_model.Data['InputVars'].loc[case_id, :, '{}_Inflow'.format(p_type)] = last_year_mean_tp * percent
            else:
                if last_year_mean_tn > lake_water_quality_target_df.loc[case_id, 'TN_Target']:
                    yearly_tn_reduction = (last_year_mean_tn - lake_water_quality_target_df.loc[case_id, 'TN_Target']) / len(range(2023, 2051))
                    for year in range(2023, 2051):
                        last_year_mean_tn -= yearly_tn_reduction
                        for n_type, percent in Initial_distribute_dics['TN'].items():
                            calibrated_model.Data['InputVars'].loc[case_id, '{}-1-1'.format(year): '{}-12-31'.format(year), '{}_Inflow'.format(n_type)] = last_year_mean_tn * percent
                else:
                    for n_type, percent in Initial_distribute_dics['TN'].items():
                        calibrated_model.Data['InputVars'].loc[case_id, :, '{}_Inflow'.format(n_type)] = last_year_mean_tn * percent

                if last_year_mean_tp > lake_water_quality_target_df.loc[case_id, 'TP_Target']:
                    yearly_tp_reduction = (last_year_mean_tp - lake_water_quality_target_df.loc[case_id, 'TP_Target']) / len(range(2023, 2051))
                    for year in range(2023, 2051):
                        last_year_mean_tp -= yearly_tp_reduction
                        for p_type, percent in Initial_distribute_dics['TP'].items():
                            calibrated_model.Data['InputVars'].loc[case_id, '{}-1-1'.format(year): '{}-12-31'.format(year), '{}_Inflow'.format(p_type)] = last_year_mean_tp * percent
                else:
                    for p_type, percent in Initial_distribute_dics['TP'].items():
                        calibrated_model.Data['InputVars'].loc[case_id, :, '{}_Inflow'.format(p_type)] = last_year_mean_tp * percent

        return calibrated_model

    def run(self, save_dir):
        # Run and save
        for model_name, model in self.model_dic.items():
            save_sub_dir = os.path.join(save_dir, model_name)
            _ = None if os.path.exists(save_sub_dir) else os.mkdir(save_sub_dir)

            model.run(if_show_bar=True)
            model.save_data(
                save_dir=save_sub_dir,
                data_names=['StateVars'],
                if_show_bar=True
            )


if __name__ == '__main__':
    scenario_model = ScenarioModelOfWithoutNutrientReductionDuring2005to2022(
        dir_calibrated_paras='Results/ModelOptimizationResults'
    )
    scenario_model.run(
        save_dir='Results/ScenarioModelResultsWithoutNutrientReduction'
    )

    future_scenario_model = ScenarioModelOfFuture(
        dir_calibrated_paras='Results/ModelOptimizationResults',
        climate_scenarios=['ssp245', 'ssp585'],
        nutrient_scenarios=['NutrientReduction', 'WithoutNutrientReduction']
    )
    future_scenario_model.run(
        save_dir='Results'
    )
