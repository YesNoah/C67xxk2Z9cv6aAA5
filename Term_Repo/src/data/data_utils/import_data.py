# import necessary libraries
import pandas as pd
import os
import glob


def CSVtoDF(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    print('Location:', os.path.join(path, filename))
    return df 
      

#def multiCSVread():
    #use glob to get all the csv files 
    # in the folder
    #csv_files = glob.glob(os.path.join(path, "*.csv"))
        # loop over the list of csv files
    #for f in csv_files:
        
        # read the csv file
        #df = pd.read_csv(f)
        
        # print the location and filename
        #print('Location:', f)
        #print('File Name:', f.split("\\")[-1])
        
        # print the content
        #print('Content:\n', df.head())

    #CSVtoDF(r'C:\Users\noahc\Happy_Repo\data\raw')