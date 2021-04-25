from helpers.utils import *
from helpers.constants import Constants as constants
import pickle

DIR = constants.DIR

def prepare_data():
    download_current_data(DIR)
    train, val, tournament = load_data(DIR, reduce_memory=True)
    
    return train, tournament 