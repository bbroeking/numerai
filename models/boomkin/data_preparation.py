from helpers.utils import *
from helpers.constants import Constants as constants
from sklearn.decomposition import PCA
import pickle

DIR = constants.DIR

def prepare_data():
    # Download, unzip and load data
    download_current_data(DIR)
    train_static, val_static, tournament_static = load_data(DIR, reduce_memory=True)
    features_list = generate_features_list(train_static)
    # Group stats
    train_with_group, tournament_with_group = get_group_stats(train_static), get_group_stats(tournament_static)

    pca = PCA(0.95, svd_solver='full')
    pca.fit(train_with_group[features_list])
    res = pca.transform(train_with_group[features_list])

    train_with_group_pca = pca.transform(train_with_group[features_list])
    tournament_with_group_pca = pca.transform(tournament_with_group[features_list])

    train_with_group_pca_df = pd.DataFrame(train_with_group_pca)
    tournament_with_group_pca_df = pd.DataFrame(tournament_with_group_pca)

    pca_train = pd.concat([train_with_group.drop(columns=features_list).reset_index(),
                        train_with_group_pca_df.reset_index()], axis=1)
    pca_tournament = pd.concat([tournament_with_group.drop(columns=features_list).reset_index(),
                                tournament_with_group_pca_df.reset_index()], axis=1)

    pca_train = pca_train.drop(columns=['index'])
    pca_tournament = pca_tournament.drop(columns=['index'])

    return pca_train, pca_tournament