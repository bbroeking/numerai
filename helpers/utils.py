import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import numerapi
from scipy import stats
NAPI = numerapi.NumerAPI(verbosity="info")

from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix, mean_absolute_error

def load_model(name="model.pickle.dat"):
    return pickle.load(open(name, "rb"))
    
def save_model(model, name="model.pickle.dat"):
    return pickle.dump(model, open(name, "wb"))

# The models should be scored based on the rank-correlation (spearman) with the target
def numerai_score(y_true, y_pred):
    rank_pred = y_pred.apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

# It can also be convenient while working to evaluate based on the regular (pearson) correlation
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def download_current_data(directory: str):
    """
    Downloads the data for the current round
    :param directory: The path to the directory where the data needs to be saved
    """
    current_round = NAPI.get_current_round()
    if os.path.isdir(f'{directory}/numerai_dataset_{current_round}/'):
        print(f"You already have the newest data! Current round is: {current_round}")
    else:
        print(f"Downloading new data for round: {current_round}!")
        NAPI.download_current_dataset(dest_path=directory, unzip=True)

def load_data(directory: str, reduce_memory: bool=True) -> tuple:
    """
    Get data for current round
    :param directory: The path to the directory where the data needs to be saved
    :return: A tuple containing the datasets
    """
    print('Loading the data')
    full_path = f'{directory}/numerai_dataset_{NAPI.get_current_round()}/'
    train_path = full_path + 'numerai_training_data.csv'
    test_path = full_path + 'numerai_tournament_data.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # Reduce all features to 32-bit floats
    if reduce_memory:
        num_features = [f for f in train.columns if f.startswith("feature")]
        train[num_features] = train[num_features].astype(np.float32)
        test[num_features] = test[num_features].astype(np.float32)
    val = test[test['data_type'] == 'validation']
    tournament = test
    return train, val, tournament

def get_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
        cols = [col for col in df.columns if group in col]
        df.loc[:, f"feature_{group}_mean"] = df[cols].mean(axis=1)
        df.loc[:, f"feature_{group}_std"] = df[cols].std(axis=1)
        df.loc[:, f"feature_{group}_skew"] = df[cols].skew(axis=1)
    return df

def sharpe_ratio(corrs: pd.Series) -> np.float32:
    """
    Calculate the Sharpe ratio for Numerai by using grouped per-era data

    :param corrs: A Pandas Series containing the Spearman correlations for each era
    :return: A float denoting the Sharpe ratio of your predictions.
    """
    return corrs.mean() / corrs.std()

def evaluate(df: pd.DataFrame) -> tuple:
    """
    Evaluate and display relevant metrics for Numerai 

    :param df: A Pandas DataFrame containing the columns "era", "target" and a column for predictions
    :param pred_col: The column where the predictions are stored
    :return: A tuple of float containing the metrics
    """
    def _score(sub_df: pd.DataFrame) -> np.float32:
        """Calculates Spearman correlation"""
        return stats.spearmanr(sub_df["target"], sub_df["prediction"])[0]

    # Calculate metrics
    corrs = df.groupby("era").apply(_score)
    print(corrs)
    payout_raw = (corrs / 0.2).clip(-1, 1)
    spearman = round(corrs.mean(), 4)

    payout = round(payout_raw.mean(), 4)
    numerai_sharpe = round(sharpe_ratio(corrs), 4)
    mae = mean_absolute_error(df["target"], df["prediction"]).round(4)

    # Display metrics
    print(f"Spearman Correlation: {spearman}")
    print(f"Average Payout: {payout}")
    print(f"Sharpe Ratio: {numerai_sharpe}")
    print(f"Mean Absolute Error (MAE): {mae}")
    return spearman, payout, numerai_sharpe, mae    

def _neutralize(df, columns, by, proportion=1.0):
    scores = df[columns]
    exposures = df[by].values
    scores = scores - proportion * exposures.dot(numpy.linalg.pinv(exposures).dot(scores))
    return scores / scores.std(ddof=0)

def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)

def normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]

def generate_features_list(df):
    return [c for c in df if str(c).startswith("feature")]

def clean_for_xgboost(df, dropped_columns=None):
    if dropped_columns is None:
        dropped_columns = ['id', 'era', 'data_type', 'target']

    return df.drop(dropped_columns, axis=1), df['target']

def generate_preds(model, df, dtrain):
    df.loc[:, "prediction"] = model.predict(dtrain)
    return df

def generate_submission(df, name="submission.csv"):
    by = pd.read_csv('data/numerai_dataset_'+str(NAPI.get_current_round())+'/example_predictions.csv')
    submission = pd.concat([by.drop(columns='prediction'), df['prediction']], axis=1)
    submission = submission.set_index('id')
    submission.to_csv(name)

def neutralize_series(series, by, proportion=1.0):
   scores = series.values.reshape(-1, 1)
   exposures = by.values.reshape(-1, 1)

   # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
   exposures = np.hstack(
       (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

   correction = proportion * (exposures.dot(
       np.linalg.lstsq(exposures, scores)[0]))
   corrected_scores = scores - correction
   neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
   return neutralized

def generate_polynomial_features(feature_list, train, tournament):
    """
    :param feature_list: list of features to generate with
    :param train: train data
    :param tournament: tournament data
    """
    interactions = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    # These can be condensed into fit_transform
    interactions.fit(train[feature_list], train["target"])
    X_train_interact = pd.DataFrame(interactions.transform(train[feature_list]))
    X_best_tournament_inter = pd.DataFrame(interactions.transform(tournament[feature_list]))

    train = pd.concat([train, X_train_interact],axis=1)
    tournament = tournament.reset_index().drop(columns='index')
    tournament = pd.concat([tournament, X_best_tournament_inter],axis=1)

    return train, tournament

def clean_era(df):
    df['era'] = df.loc[:, 'era'].str[3:].astype('int32')
    return df

def setup_xgboost_training(train):
    X, y = clean_for_xgboost(train)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    dtrain = xgboost.DMatrix(X_train, y_train)
    dtest = xgboost.DMatrix(X_test, y_test)
    return dtrain, dtest

def ar1(x):
    return np.corrcoef(x[:-1], x[1:])[0,1]

def autocorr_penalty(x):
    n = len(x)
    p = ar1(x)
    return np.sqrt(1 + 2*np.sum([((n - i)/n)*p**i for i in range(1,n)]))

def smart_sharpe(x):
    return np.mean(x)/(np.std(x, ddof=1)*autocorr_penalty(x))