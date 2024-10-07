from classes import *
from utils import *
from multiprocessing import Pool, Manager


def train_with_config(config):
    architecture = config["architecture"]
    for mode in proc_params['modes']:
        for divergence in proc_params['divergences']:
            if divergence == "CPC" and ("deranged" in architecture):
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = 0
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = 0
            else:
                start_time = time.time()
                est_mi = fDIME(proc_params, divergence, architecture, mode)
                mi_estimates = np.zeros(proc_params['len_step'] * len(proc_params['levels_MI']))
                for idx, level_MI in enumerate(proc_params['levels_MI']):
                    est_mi.update_SNR_or_rho(level_MI)
                    est_mi_training_estimates_tmp = est_mi.train(epochs=proc_params['len_step'], batch_size=opt_params['batch_size'])
                    mi_estimates[proc_params['len_step'] * idx:proc_params['len_step'] * (idx + 1)] = est_mi_training_estimates_tmp
                end_time = time.time()
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = (float(end_time) - float(start_time)) / 60
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = mi_estimates


def train_unif_with_config(config):
    architecture = config["architecture"]
    for mode in proc_params['modes']:
        for divergence in proc_params['divergences']:
            if divergence == "CPC" and ("deranged" in architecture):
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = 0
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = 0
            else:
                start_time = time.time()
                est_mi = fDIME(proc_params, divergence, architecture, mode)
                mi_estimates = np.zeros(proc_params['len_step'] * len(proc_params['levels_eps']))
                for idx, level_eps in enumerate(proc_params['levels_eps']):
                    est_mi.update_eps_unif(level_eps)
                    est_mi_training_estimates_tmp = est_mi.train(epochs=proc_params['len_step'], batch_size=opt_params['batch_size'])
                    mi_estimates[proc_params['len_step'] * idx:proc_params['len_step'] * (idx + 1)] = est_mi_training_estimates_tmp
                end_time = time.time()
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = (float(end_time) - float(start_time)) / 60
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = mi_estimates


def train_swiss_with_config(config):
    architecture = config["architecture"]
    for mode in proc_params['modes']:
        for divergence in proc_params['divergences']:
            if divergence == "CPC" and ("deranged" in architecture):
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = 0
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = 0
            else:
                start_time = time.time()
                est_mi = fDIME(proc_params, divergence, architecture, mode)
                mi_estimates = np.zeros(proc_params['len_step'] * len(proc_params['levels_MI']))
                for idx, level_MI in enumerate(proc_params['levels_MI']):
                    est_mi.update_SNR_or_rho(level_MI)
                    est_mi_training_estimates_tmp = est_mi.train(epochs=proc_params['len_step'], batch_size=opt_params['batch_size'])
                    mi_estimates[proc_params['len_step'] * idx:proc_params['len_step'] * (idx + 1)] = est_mi_training_estimates_tmp
                end_time = time.time()
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = (float(end_time) - float(start_time)) / 60
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = mi_estimates


def train_student_with_config(config):
    architecture = config["architecture"]
    for mode in proc_params['modes']:
        for divergence in proc_params['divergences']:
            if divergence == "CPC" and ("deranged" in architecture):
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = 0
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = 0
            else:
                start_time = time.time()
                est_mi = fDIME(proc_params, divergence, architecture, mode)
                mi_estimates = np.zeros(proc_params['len_step'] * len(proc_params['levels_df']))
                for idx, level_df in enumerate(proc_params['levels_df']):
                    est_mi.update_df(level_df)
                    est_mi_training_estimates_tmp = est_mi.train(epochs=proc_params['len_step'], batch_size=opt_params['batch_size'])
                    mi_estimates[proc_params['len_step'] * idx:proc_params['len_step'] * (idx + 1)] = est_mi_training_estimates_tmp
                end_time = time.time()
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = (float(end_time) - float(start_time)) / 60
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = mi_estimates


if __name__ == '__main__':

    scenario = "student"  # uniform  staircase student

    if scenario == "staircase":
        batch_sizes = [64]
        architectures_list = [["joint", "deranged"]]
        for batch_size in batch_sizes:
            for architectures in architectures_list:
                proc_params = {
                    'levels_MI': [2,4,6,8,10],
                    'len_step': 4000,
                    'alpha': 1,
                    'latent_dim': 5,
                    'divergences': ["KL", "HD", "GAN"],
                    'architectures': architectures,
                    'tot_len_stairs': 20000,
                    'modes': ["gauss", "cubic", "asinh", "half-cube"],
                    'rho_gauss_corr': False,
                    'device': "cpu"
                }
                opt_params = {
                    'batch_size': batch_size
                }

                training_configs = [
                    {"architecture": architectures[0]},
                    {"architecture": architectures[1]},
                ]

                manager = Manager()
                staircases = manager.dict()
                time_dict = manager.dict()
                pool = Pool(processes=2)
                pool.map(train_with_config, training_configs)
                pool.close()
                print("staircases: ", staircases)
                plot_staircases(staircases, proc_params, opt_params, proc_params['latent_dim'])
                save_time_dict(time_dict, proc_params['latent_dim'], opt_params["batch_size"], proc_params, scenario)

    elif scenario == "uniform":
        batch_sizes = [64]
        architectures_list = [["joint"]]
        for batch_size in batch_sizes:
            for architectures in architectures_list:
                proc_params = {
                    'levels_eps': [2, 1, 1/2, 1/10, 1/20],
                    'levels_MI': None,
                    'rho_gauss_corr': None,
                    'len_step': 4000,
                    'alpha': 1,
                    'latent_dim': 1,
                    'divergences': ["KL", "HD", "GAN"],
                    'architectures': architectures,
                    'tot_len_stairs': 20000,
                    'modes': ["uniform"],
                    'device': "cpu"
                }
                opt_params = {
                    'batch_size': batch_size
                }

                training_configs = [
                    {"architecture": architectures[0]}
                ]

                manager = Manager()
                staircases = manager.dict()
                time_dict = manager.dict()
                pool = Pool(processes=2)
                pool.map(train_unif_with_config, training_configs)
                pool.close()
                print("staircases: ", staircases)
                plot_staircases_unif(staircases, proc_params, opt_params, proc_params['latent_dim'])
                save_time_dict(time_dict, proc_params['latent_dim'], opt_params["batch_size"], proc_params, scenario)

    elif scenario == "swiss":
        batch_sizes = [64]
        architectures_list = [["joint"]]
        for batch_size in batch_sizes:
            for architectures in architectures_list:
                proc_params = {
                    'levels_MI': [0.8, 2, 3],
                    'rho_gauss_corr': False,
                    'len_step': 4000,
                    'alpha': 1,
                    'latent_dim': 1,
                    'divergences': ["KL", "GAN"],
                    'architectures': architectures,
                    'tot_len_stairs': 12000,
                    'modes': ["swiss"],
                    'device': "cpu"
                }
                opt_params = {
                    'batch_size': batch_size
                }

                training_configs = [
                    {"architecture": architectures[0]}
                ]

                manager = Manager()
                staircases = manager.dict()
                time_dict = manager.dict()
                pool = Pool(processes=2)
                pool.map(train_swiss_with_config, training_configs)
                pool.close()
                print("staircases: ", staircases)
                plot_staircases_swiss(staircases, proc_params, opt_params, proc_params['latent_dim'])
                save_time_dict(time_dict, proc_params['latent_dim'], opt_params["batch_size"], proc_params, scenario)

    elif scenario == "student":
        batch_sizes = [64]
        architectures_list = [["joint"]]
        for batch_size in batch_sizes:
            for architectures in architectures_list:
                proc_params = {
                    'rho_gauss_corr': False,
                    'levels_df': [3, 2, 1],
                    'len_step': 4000,
                    'alpha': 1,
                    'latent_dim': 5,
                    'divergences': ["KL", "GAN"],
                    'architectures': architectures,
                    'tot_len_stairs': 12000,
                    'modes': ["student"],
                    'device': "cpu"
                }
                opt_params = {
                    'batch_size': batch_size
                }

                training_configs = [
                    {"architecture": architectures[0]}
                ]

                manager = Manager()
                staircases = manager.dict()
                time_dict = manager.dict()
                pool = Pool(processes=2)
                pool.map(train_student_with_config, training_configs)
                pool.close()
                print("staircases: ", staircases)
                plot_staircases_student(staircases, proc_params, opt_params, proc_params['latent_dim'])
                save_time_dict(time_dict, proc_params['latent_dim'], opt_params["batch_size"], proc_params, scenario)

