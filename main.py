import libemg
from utils import prepare_data
    

    





if __name__ == "__main__":
    dataset_dir = 'dataset/'
    # prepare means get it into a format that libemg can consume it -- 
    # a collection of .csv files
    prepare_data(dataset_dir)