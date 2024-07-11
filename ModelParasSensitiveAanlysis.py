import matplotlib.pyplot as plt
import pandas as pd
from SALib.sample import morris
from SALib.analyze import morris as moa
from copy import deepcopy
from NationalHABModel import Model
from Config import *
from Utils.MutilProcess import mutil_process
from tqdm import tqdm
import gc
from sklearn.preprocessing import MinMaxScaler


class ModelParasSensitiveAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def morris_train(start_repeat, save_dir, sample_number=1, num_levels=4):
        problem = ModelParasSensitiveAnalysis.load_problem('L2')

        # shape = [(num_vars + 1) * sample_number, num_vars]
        samples = morris.sample(problem=problem, N=sample_number, num_levels=num_levels)

        t = Model(
            input_dir='Data/ModelInput',
            case_ids=list(UsedCases_dic.keys()),
            n_repeat=samples.shape[0],
            start_repeat=start_repeat,
            deal_range_out='clip'
        )

        t.Data['Paras'].loc[:, :, problem['names']] = np.repeat(samples[np.newaxis, :, :], len(t.case_ids), axis=0)

        t.run(if_show_bar=True)

        t.save_data(save_dir=save_dir, data_names=['StateVars', 'Paras'], if_show_bar=True)
        del t
        gc.collect()

    @staticmethod
    def morris_analysis(save_dir, repeat_range):

        Ys = ['Chla', 'TN', 'TP']
        si_dic = {
            y: pd.DataFrame(
                index=list(UsedCases_dic.keys())
            )
            for y in Ys
        }

        for case_id, case_name in UsedCases_dic.items():
            sample_paras_df = pd.DataFrame(
                index=repeat_range
            )
            y_df = pd.DataFrame(
                index=repeat_range
            )
            for repeat in repeat_range:
                paras_path = os.path.join(save_dir, '{}-{}-Paras-{}.csv'.format(case_id, case_name, repeat))
                paras_df = pd.read_csv(paras_path, index_col=0)
                sample_paras_df.loc[repeat, paras_df.index] = paras_df.loc[:, 'Value'].values

                sv_path = os.path.join(save_dir, '{}-{}-StateVars-{}.csv'.format(case_id, case_name, repeat))
                sv_df = pd.read_csv(sv_path, index_col=0)
                for y in Ys:
                    y_df.loc[repeat, y] = sv_df.loc[:, y].mean()

            for y in Ys:
                moa_result_dic = moa.analyze(
                    problem=ModelParasSensitiveAnalysis.load_problem('L2'),
                    X=sample_paras_df.values,
                    Y=y_df.loc[:, y].values
                )
                si_dic[y].loc[case_id, sample_paras_df.columns] = moa_result_dic['mu_star']
            print(case_id)

        for y in Ys:
            si_dic[y].to_csv(os.path.join(save_dir, '0-PSA-{}.csv'.format(y)))

    @staticmethod
    def plt_morris_results(load_dir, save_path):
        problem = ModelParasSensitiveAnalysis.load_problem('L2')
        paras_dic = {
            'Chla': [p for p in problem['names'][: 46] if p != 'Rate_Deposition_DO'],
            'TN': [p for p in problem['names'][76:] if 'ON' in p or 'NH3' in p or 'NOx' in p],
            'TP': [p for p in problem['names'][76:] if 'OP' in p or 'PO4' in p],
        }
        scale_dic = {
            'Chla': 300,
            'TN': 1,
            'TP': 10,
        }
        text_ls = [
            'a. Parameters in phytoplankton dynamics',
            'b. Parameters in nitrogen cycles',
            'c. Parameters in phosphorus cycles',
        ]
        lw = 0.25


        plt.figure(figsize=(14, 7))
        for i, y in enumerate(paras_dic.keys()):
            if i == 0:
                ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            else:
                ax = plt.subplot2grid((2, 2), (1, i-1))

            plt.sca(ax)
            paras_names = paras_dic[y]
            si_df = pd.read_csv(os.path.join(load_dir, '0-PSA-{}.csv'.format(y)), index_col=0).loc[:, paras_names] * scale_dic[y]
            median_arr = si_df.median(axis=0).values
            decrease_index = np.argsort(-median_arr)
            si_df = si_df.loc[:, [si_df.columns[idx] for idx in decrease_index]]

            plt.boxplot(
                si_df.values,
                positions=range(len(paras_names)),
                medianprops={'color': 'k', 'linewidth': lw * 2},
                flierprops={'markeredgewidth': lw, 'markersize': 4},
                widths=0.5,
                boxprops={'linewidth': lw * 2},
                whiskerprops={'linewidth': lw * 2},
                capprops={'linewidth': lw * 2}
            )

            x_ticklabels = list(si_df.columns)
            x_ticklabels = list(map(lambda x:  x.replace('Rate', 'K')
                                                .replace('HSC_C', 'Ref')
                                                .replace('HSC_P', 'HKP')
                                                .replace('HSC_N', 'HKN')
                                                .replace('HSC_O', 'HKO')
                                                .replace('Opt_T', 'OptT')
                                                .replace('Opt_SR', 'OptI')
                                                .replace('Excretion', 'Excr')
                                                .replace('Predation', 'Pred')
                                                .replace('Mortality', 'Mort')
                                                .replace('Settling', 'Settl')
                                                .replace('Photosynthesis', 'Phot')
                                                .replace('Respiration', 'Resp')
                                                .replace('Resuspension', 'Resu')
                                                .replace('Releasing', 'Rele')
                                                .replace('Nitrification', 'Nit')
                                                .replace('Denitrification', 'Denit')
                                                .replace('Mineralization', 'Mine')
                                                .replace('Adsorption', 'Adso')
                                                .replace('Anammox', 'Anam')
                                                .replace('Deposition', 'Depo')
                                    , x_ticklabels))

            plt.xticks(range(len(paras_names)), x_ticklabels, rotation=270)

            if i == 0:
                plt.xlim([-1, 45])
            else:
                plt.xlim([-1, 20])

            plt.text(0.01, 0.9, text_ls[i], transform=ax.transAxes, fontsize=12)
            ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=10)

            for spine in ax.spines.values():
                spine.set_linewidth(lw)

            plt.ylim([0, 1])
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            plt.grid(which='major', axis='y', color='grey', linestyle='--', linewidth=0.25, alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300.)

    @staticmethod
    def load_problem(case_id):
        paras_path = 'Data/ModelInput/{}-{}-Paras.csv'.format(case_id, AllCases_dic[case_id])
        paras_df = pd.read_csv(paras_path, index_col=0)

        paras_names = list(paras_df.index)
        bounds = [[paras_df.loc[paras_name, 'RangeMin'], paras_df.loc[paras_name, 'RangeMax']] for paras_name in paras_names]

        problem = {
            'num_vars': len(paras_names),
            'names': paras_names,
            'bounds': bounds,
        }
        return problem


if __name__ == '__main__':
    repeat_onetime = 100
    n_repeat = 10
    save_dir = 'Results/ModelParasSensitiveAnalysis'
    save_path = os.path.join(save_dir, 'PSA.png')

    for start_repeat in [i * repeat_onetime for i in range(n_repeat)]:
        ModelParasSensitiveAnalysis.morris_train(
            start_repeat=start_repeat,
            save_dir=save_dir
        )

    ModelParasSensitiveAnalysis.morris_analysis(save_dir, range(repeat_onetime * n_repeat))
