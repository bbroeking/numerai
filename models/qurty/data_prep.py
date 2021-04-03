from helpers.utils import *
import pickle

DIR = "data"
DEFAULT_FT_CORR_LIST = 'qurty_ft_corr_list.pickle.dat'

def prepare_data(ft_corr_list=None):
    download_current_data(DIR)
    train_static, val_static, tournament_static = load_data(DIR, reduce_memory=True)
    features_list = generate_features_list(train_static)

    train_with_group = get_group_stats(train_static)
    tournament_with_group = get_group_stats(tournament_static)

    if ft_corr_list is None:
        ft_corr_list = DEFAULT_FT_CORR_LIST

    with open(ft_corr_list,'rb') as f:
        ft_corr = pickle.load(f)

    train, tournament = generate_polynomial_features(ft_corr, train_with_group, tournament_with_group)

    return train, tournament 