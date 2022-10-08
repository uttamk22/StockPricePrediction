#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:43:41 2022

@author: uttamkedia

"""
import pandas as pd
from glob import glob
import requests
import os
import time
from tqdm import tqdm
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm


def tech_file_load(tickers, technical_file):
    
    #read the technical file if exists and get the list of stocks data present
    if os.path.exists(technical_file):
        tech_file_df = pd.read_csv(technical_file)
        stock_codes_loaded = tech_file_df['stock code'].tolist()
        
    else:
        stock_codes_loaded =  []
    stock_code_remains = [x for x in tickers if x not in stock_codes_loaded]
    #if any stock download pending then download the weekly data through API alphavantage 
    if stock_code_remains:
        for ticker in tqdm(stock_code_remains, desc="Fetching Stock Technical Data through API"):
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={ticker}.BSE&outputsize=full&apikey==KU5KBFD34SV59FIR&datatype=csv"
            technical_df = pd.read_csv(url)
            technical_df.insert(loc=0, column='stock code', value= ticker)
            #create the file if doesn't exist or append the new stock data 
            if os.path.exists(technical_file):
                technical_df.to_csv(technical_file, mode='a', index=False, header=False)
            else:
                technical_df.to_csv(technical_file, mode='w', index=False)
            time.sleep(15)
    else:
        print("Technical Weekly Data Downloading Completed")
        
        
if __name__ == "__main__":
    fundamemtal_file = '/Users/Uttamkedia/Downloads/bse_stocks_fundamental_yearly_data.csv'
    technical_file = '/Users/Uttamkedia/Downloads/bse_stocks_technical_weekly_data_latest.csv'
#tickers = fund_file_load(stock_files,fundamemtal_file)

    funda_file_df = pd.read_csv(fundamemtal_file)
    tickers = funda_file_df['Stock code'].tolist()
    print(tickers)
    tech_file_load(tickers, technical_file)