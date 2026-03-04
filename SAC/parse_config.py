import json


def load_settings(config_path):

    with open(config_path, 'r') as file:
        data = json.load(file)

    training_data = data['Training']
    SAC_settings = training_data['SAC']
    episodes = training_data['episodes']
    env = training_data['env']

    # LOADING SAC SETTINGS
    if SAC_settings['temp'] == 'fixed':
        temp_val = SAC_settings['temp_val']
        fixed = True
    elif SAC_settings['temp'] == 'learnt':
        temp_val = 1.0
        fixed = False

    batch_size = SAC_settings['batch']
    buffer_size = SAC_settings['buffer']
    k_future = SAC_settings['k_future']
    sac_tau = SAC_settings['tau']
    sac_gamma = SAC_settings['gamma']
    sac_lr = SAC_settings['learn_rate']

    # Intrinsic Motivation settings
    # TBC

    return env, episodes, (fixed, temp_val, batch_size, buffer_size ,k_future, sac_tau, sac_gamma, sac_lr)
