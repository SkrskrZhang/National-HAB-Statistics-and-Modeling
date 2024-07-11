import numpy as np
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import pymc as pm
from multiprocessing import freeze_support
import arviz as az
import pandas as pd
from tqdm import tqdm
import xarray as xr
import time, os
import psutil
from multiprocessing import Process, current_process
from functools import reduce
import sys

from NationalHABModel import Model
from Config import UsedCases_dic


# 5 in phytoplankton module + 3 in nitrogen module + 2 in phorsphorus module
optimized_paras_ls = [
    'Rate_Grow_Fb', 'Rate_Grow_Fd', 'Rate_Grow_Fg', 'HSC_P_Grow_Fb', 'Opt_T_Grow_Fb',
    'Rate_Settling_ON', 'Rate_Releasing_NH3', 'Rate_Denitrification_NOx',
    'Rate_Settling_OP', 'Rate_Releasing_PO4',
]


class CustomFunction(pt.Op):
    itypes = [pt.dvector, pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dvector]  # outputs a single scalar value (the log likelihood)

    def __init__(self, x, group_indicator):
        self.x = x
        self.group_indicator = group_indicator
        self.n_iter = 0

    def perform(self, node, inputs, outputs):
        # Evaluate the Op result for a specific numerical input

        # The inputs are always wrapped in a list
        (input0, input1,) = inputs
        result = self.my_model(input0, input1)
        # The results should be assigned inplace to the nested list
        # of outputs provided by PyTensor. If you have multiple
        # outputs and results, you should assign each at outputs[i][0]
        outputs[0][0] = np.asarray(result, dtype=np.float64)

    def my_model(self, input0, input1):
        date_range = pd.date_range(start='2020-1-1', end='2020-12-31')
        df = pd.DataFrame(index=date_range)
        pbar = tqdm(total=len(date_range), ncols=100, desc='Run iter {} in core {}'.format(self.n_iter, current_process().name.split("_")[-1]))
        for date in date_range:
            _input0 = np.log(np.exp(input0)) + input0 - input0 * 0.1 * 10.
            _input1 = np.log(np.exp(input1)) + input1 - input1 * 0.1 * 10.
            df.loc[date, 'a'] = _input0[0]
            pbar.update()
        pbar.close()
        # input0[0] = df['a'].mean()
        self.n_iter += 1
        return input1[self.group_indicator] * self.x + input0[self.group_indicator]


class MyModel(pt.Op):
    itypes = [pt.dvector, pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dvector]  # outputs a single scalar value (the log likelihood)

    def __init__(self, case_ids, paras_ls, n_chains, target, mea_df, save_dir):
        self.case_ids = case_ids
        self.paras_ls = paras_ls
        self.n_chains = n_chains
        self.target = target
        self.mea_df = mea_df
        self.Model_dic = {
            i: Model(
                input_dir='Data/ModelInput',
                case_ids=case_ids,
                n_repeat=1,
                start_repeat=0,
                deal_range_out='clip'
            )
            for i in range(n_chains)
        }

        self.save_dir_dic = {
            i: os.path.join(save_dir, str(i))
            for i in range(n_chains)
        }
        _ = [os.mkdir(save_dir) for save_dir in self.save_dir_dic.values() if not os.path.exists(save_dir)]

        self.n_iter = 0

    def perform(self, node, inputs, outputs):
        # Evaluate the Op result for a specific numerical input

        # The inputs are always wrapped in a list
        result = self.my_model(*inputs)
        # The results should be assigned inplace to the nested list
        # of outputs provided by PyTensor. If you have multiple
        # outputs and results, you should assign each at outputs[i][0]
        outputs[0][0] = np.asarray(result, dtype=np.float64)

    def my_model(self, *args):
        """
        Calc the model and get the results compared to mea data
        :param input0: Numpy Array with shape (n_groups, )
        :param input1: Numpy Array with shape (n_groups, )
        :return:
        """
        # Get the model according to core index
        core_name = current_process().name.split("_")[-1]
        if core_name == 'MainProcess':
            core_idx = 0
            model = self.Model_dic[core_idx]
        else:
            core_idx = int(core_name)
            model = self.Model_dic[core_idx]

        # Load the paras from args
        for paras_idx, paras_arr in enumerate(args):
            for case_idx, case_id in enumerate(self.case_ids):
                model.Data['Paras'].loc[case_id, 0, self.paras_ls[paras_idx]] = paras_arr[case_idx]

        # Run the model
        if core_name == 'MainProcess':
            model.load_data(
                load_dir=self.save_dir_dic[0],
                if_show_bar=True,
                load_data_types=['Flux', 'StateVars', 'Paras']
            )
        else:
            model.run(if_show_bar=True)

        # Save the results and get the simulated data compared to mea data.
        model.set_start_repeat(self.n_iter)
        model.save_data(
            save_dir=self.save_dir_dic[core_idx],
            data_names=['Flux', 'StateVars', 'Paras'],
            if_show_bar=False
        )
        model.set_start_repeat(0)

        self.n_iter += 1
        return self.get_sim_data(model)

    def get_sim_data(self, model):
        """
        Get the simulated data compared to mea date
        :return: Numpy Array with the shape (n, ), the n must equal to mea.shape[0]
        """
        for case_id in self.case_ids:
            case_mea_df = self.mea_df[self.mea_df['CaseID'] == case_id]
            self.mea_df.loc[case_mea_df.index, 'Sim'] = model.Data['StateVars'].loc[case_id, 0, pd.to_datetime(case_mea_df['Date'].values), self.target].values
        return self.mea_df['Sim'].values


class ModelOptimizerHierarchicalBayesian:
    def __init__(self, case_ids, target):
        self.case_ids = case_ids
        self.target = target
        self.compare_type = 'mean'
        self.ini_model = Model(
            input_dir='Data/ModelInput',
            case_ids=list(UsedCases_dic.keys()),
            n_repeat=1,
            start_repeat=0,
            deal_range_out='clip'
        )

    def hierarchical_bayes(self, save_dir, n_chains=1, n_samples=1000):
        mea_df = self.get_mea_data()

        custom_function = MyModel(
            case_ids=self.case_ids,
            paras_ls=optimized_paras_ls,
            n_chains=n_chains,
            target=self.target,
            mea_df=mea_df,
            save_dir=save_dir,
        )

        this = sys.modules[__name__]

        with pm.Model() as hierarchical_model:
            tensor_ls = []
            for paras in optimized_paras_ls:
                # Hyperpriors (Normal dist)
                ini_value = self.ini_model.Data['Paras'].loc[self.case_ids, 0, paras].values.mean()
                setattr(this, 'mu_{}'.format(paras), pm.Normal('mu_{}'.format(paras), mu=ini_value, sigma=ini_value))
                setattr(this, 'sigma_{}'.format(paras), pm.HalfNormal('sigma_{}'.format(paras), sigma=ini_value * 0.5))

                # Lake specific priors (Normal dist)
                setattr(
                    this,
                    paras,
                    pm.Normal(
                        paras, mu=locals()['mu_{}'.format(paras)],
                        sigma=locals()['sigma_{}'.format(paras)],
                        shape=len(self.case_ids),
                    )
                )

                # Load paras to model and get simulated data
                setattr(this, 'tv_{}'.format(paras), pt.as_tensor_variable(locals()[paras]))

                tensor_ls.append(locals()['tv_{}'.format(paras)])

            mu = custom_function(*tensor_ls)
            simulatedSigma = pm.HalfNormal('simulatedSigma', sigma=0.01)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=simulatedSigma, observed=mea_df['Value'].values)

            # Sampling
            trace = pm.sample(n_samples, chains=n_chains, cores=n_chains)

        # Trace
        az.plot_trace(trace)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Hierarchical bayes trace.png'), dpi=300)
        df = trace.to_dataframe()
        df.to_csv(os.path.join(save_dir, 'Hierarchical bayes trace.csv'))

    def get_mea_data(self):
        """
        Get the mea data from all case_ids
        Return a DF with columns ['CaseID', 'Date', 'Value']
        """
        model = Model(
            input_dir='Data/ModelInput',
            case_ids=self.case_ids,
            n_repeat=1,
            start_repeat=0,
            deal_range_out='break'
        )
        mea_data = model.Data['MeaData']

        ls = []
        for case_id in self.case_ids:
            column_name = '{}_{}_{}'.format(case_id, self.compare_type, self.target)
            if column_name not in mea_data.columns:
                column_name = [c for c in mea_data.columns if c.endswith(self.target) and c.startswith('{}_'.format(case_id))][0]
            case_mea_df = mea_data[[column_name]].dropna(axis=0, how='all')
            ls.append(pd.DataFrame(data={'CaseID': [case_id] * case_mea_df.shape[0], 'Date': case_mea_df.index, 'Value': case_mea_df[column_name].values}))
        merged_df = pd.concat(ls).reset_index(drop=True)
        return merged_df

    @staticmethod
    def estimate_paras_distribution(load_dir):
        optimize_df = pd.read_csv(os.path.join(load_dir, 'Hierarchical bayes trace.csv'), header=0, index_col=0)
        optimized_paras_mean_df = optimize_df.loc[-100:, optimized_paras_ls].mean(axis=0)
        optimized_paras_std_df = optimize_df.loc[-100:, optimized_paras_ls].std(axis=0)
        optimized_paras_mean_df.to_csv(os.path.join(load_dir, 'Last100_ParasMean.csv'), index_label='Index')
        optimized_paras_std_df.to_csv(os.path.join(load_dir, 'Last100_ParasStd.csv'), index_label='Index')


def load_calibrated_paras(model, load_dir):
    # Load calibrated paras values
    optimized_paras_mean_df = pd.read_csv(os.path.join(load_dir, 'Last100_ParasMean.csv'), index_col=0, header=0)
    model.Data['Paras'].loc[optimized_paras_mean_df.index, 0, optimized_paras_ls] = optimized_paras_mean_df.loc[:, optimized_paras_ls].values
    return model


def calibrated_national_phytoplankton_module_run(load_dir):
    # Load model
    model = Model(
        input_dir='Data/ModelInput',
        case_ids=list(UsedCases_dic.keys()),
        n_repeat=1,
        start_repeat=0,
        deal_range_out='clip'
    )

    # Load calibrated paras values
    calibrated_model = load_calibrated_paras(model, load_dir)

    # Run and save
    calibrated_model.run(if_show_bar=True)
    calibrated_model.save_data(
        save_dir='Results/CalibratedModelSimulationResults',
        data_names=['StateVars'],
        if_show_bar=True
    )


if __name__ == '__main__':
    freeze_support()

    t = ModelOptimizerHierarchicalBayesian(
        case_ids=list(UsedCases_dic.keys()),
        target='Chla',
    )
    t.hierarchical_bayes(
        n_chains=5,
        n_samples=5000,
        save_dir='Results/ModelOptimizationResults'
    )
