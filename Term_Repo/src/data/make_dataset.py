# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import importlib
import data
import data.data_utils
importlib.reload(data.data_utils)
import data.data_utils.import_data as import_data
import pandas as pd

def main(path, filename, outputfilepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    df=import_data.CSVtoDF(path, filename)
    df.head()
    from data.data_utils import processing
    inputs, target, dfcsv= processing.prepro(df)

    dfcsv.to_csv(outputfilepath)

    X_train, X_test, y_train, y_test = processing.split_balance(inputs, target, dfcsv)
    return(X_train, X_test, y_train, y_test)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
