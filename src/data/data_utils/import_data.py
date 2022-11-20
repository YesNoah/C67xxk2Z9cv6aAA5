# import necessary libraries
import pandas as pd
import os
import glob


def CSVtoDF(path=r'O:\Term_Repo\data\raw'):

    #use glob to get all the csv files 
    # in the folder
    path
    #csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    df = pd.read_csv(os.path.join(path, "term-deposit-marketing-2020.csv"))
    # loop over the list of csv files
    #for f in csv_files:
        
        # read the csv file
        #df = pd.read_csv(f)
        
        # print the location and filename
        #print('Location:', f)
        #print('File Name:', f.split("\\")[-1])
        
        # print the content
        #print('Content:\n', df.head())
    print('Location:', os.path.join(path, "term-deposit-marketing-2020.csv"))
    
    return df 
      
        

#CSVtoDF(r'C:\Users\noahc\Happy_Repo\data\raw')