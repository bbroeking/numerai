from helpers.utils import *
from helpers.constants import Constants as constants
import pickle

DIR = constants.DIR
DEFAULT_FT_CORR_LIST="burningcrusade_features.pkl"

def prepare_data(ft_corr_list=None):
    download_current_data(DIR)
    train, val, tournament = load_data(DIR, reduce_memory=True)
    
    train_with_group = get_group_stats(train)
    tournament_with_group = get_group_stats(tournament)

    if ft_corr_list is None:
        ft_corr_list = DEFAULT_FT_CORR_LIST

    with open(ft_corr_list,'rb') as f:
        ft_corr = pickle.load(f)

    train, tournament = generate_polynomial_features(ft_corr, train_with_group, tournament_with_group)

    return train, tournament 