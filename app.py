#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:59:33 2022

@author: uttamkedia
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import mean_absolute_percentage_error
import time
import matplotlib.dates as mdates


#Print the instructions or source code file
def get_file_content_as_string(path):
    inputpath  = os.getcwd() + '/' + path
    try:
        with open(inputpath) as f:
            lines = f.read()
            return lines
    except:
        st.write("Sorry, Source Code is hidden now. It's a top secret project ;)")


# LOAD DATA ONCE
@st.cache()
def load_data(input_file_url):
    data = pd.read_csv(input_file_url)
    return data

#function to find out the simple moving averaged based on X(i) and Y(i) and find out the total error mse, mape
def simple_moving_average(x):
    predicted_values = []

    for i in range(len(x)):
        predicted_value = sum(x[i])/len(x[i])
        predicted_values.append(predicted_value[0])

    return predicted_values
 
#function to find out the weighted moving averaged based on X(i) and Y(i) and find out the total error mse, mape
def weighted_moving_average(x):
    predicted_values = []

    for i in range(len(x)):
        total = [w+1 for w in range(len(x[i]))]
        value = [j for j in x[i]]
        predicted_value = sum([total[e]*value[e] for e in range(len(x[i]))])/sum(total)
        predicted_values.append(predicted_value[0])

    return predicted_values

#function to find out the weighted moving averaged based on X(i) and Y(i) and find out the total error mse, mape
def exponential_moving_average(x, days,smoothing=2):
    predicted_value = x[0][0]
    predicted_values = []

    for i in range(len(x)):
        predicted_values.append(predicted_value[0])
        #print(predicted_value)  
        predicted_value = ((smoothing/(1+days))*x[i][0]) + (((1-(smoothing/(1+days)))*predicted_values[-1]))
  
    return predicted_values

#return numpy array xi and yi based on the defined moving window for dataset based on start and end index 
def univariate_window_data(dataset,moving_window, offset):
    data = []
    labels = []
    test_data = []

    start_index = moving_window
    end_index = len(dataset)

    for i in range(start_index, end_index-offset):
        indices = range(i-moving_window, i)
    # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (moving_window, 1)))
        labels.append(dataset[i+offset])
    for i in range(end_index-offset, end_index+1):
        indices = range(i-moving_window, i)
    # Reshape data from (history_size,) to (history_size, 1)
        test_data.append(np.reshape(dataset[indices], (moving_window, 1)))
       
    return np.array(data), np.array(labels) , np.array(test_data)

#preparing the dataframe x with baseline features and label y based on the window and dataset passed
def prepare_dataframe(dataset,window, offset):
    x, y , x_test = univariate_window_data(dataset,window, offset)
    columns = []
    for i in range(window):
        columns.append(f"ft_{i}")
    df = pd.DataFrame(x.reshape(len(x),x.shape[1]), columns=columns) 
    df['ema'] = exponential_moving_average(x, window)
    df['sma'] = simple_moving_average(x )
    df['wma'] = weighted_moving_average(x)
    
    df_test = pd.DataFrame(x_test.reshape(len(x_test),x_test.shape[1]), columns=columns) 
    df_test['ema'] = exponential_moving_average(x_test, window)
    df_test['sma'] = simple_moving_average(x_test)
    df_test['wma'] = weighted_moving_average(x_test)
    return (df,y, df_test)

#creating the input dataframe for all the stocks for our prediction after donwloading the data from url 
def create_input_data(input_path):
    #Data Loading through the file
    tech_df = load_data(input_path)
    #not null values 34 rows , dropping nan values
    tech_df1 = tech_df.dropna(axis=0, how = 'any', inplace = False)
    #setting date as index
    tech_df2 = tech_df1.rename(columns= {'timestamp':'tradingdate', 'stock code': 'stock_code', 'adjusted close':'adjusted_close', 'dividend amount': 'dividend_amount'}, inplace=False)
    tech_df2.index = pd.to_datetime(tech_df2.tradingdate, format='%Y-%m-%d', errors='coerce')
    #dropping column which are not correlated with the closing price 
    tech_df3 = tech_df2.drop(columns=['tradingdate', 'dividend_amount', 'volume'],inplace = False)
    return tech_df3

#filtering the input data specific to our stock    
def filter_input_data(stock_code, tech_df3):
    stock_tech_df = tech_df3[tech_df3['stock_code'] == stock_code]
    stock_uni_data_df = stock_tech_df['close']
    stock_uni_data_df = stock_uni_data_df[::-1]
    return stock_uni_data_df

#creating the test data based on the window defined
def create_test_data(stock_uni_data_df, window, offset):
    stock_uni_test_data_df = stock_uni_data_df.loc['2021-01-01':]
    index_date = stock_uni_test_data_df.index[window+offset:(stock_uni_test_data_df.index.shape[0]+window)]
    test_index_date = stock_uni_test_data_df.index[-(offset+1):]
    stock_uni_test_data= stock_uni_test_data_df.values
    x_test,y_test, x_real_test = prepare_dataframe(stock_uni_test_data,window, offset)
    return x_test,y_test, index_date, x_real_test, test_index_date

#predicting the target price based on the different models trained already
def check_model(x_test, y_test, timeframe, index_date ):

    #model = joblib.load('/Users/uttamkedia/Downloads/svr_model.joblib')
    if timeframe == 'Week':
        model_file = '/data/project12/svr_model.joblib'
    else:
        model_file = '/data/project12/svr_model_month.joblib'
    with open(model_file, 'rb') as f:
    #with open('https://github.com/uttamk22/AIML/blob/moneyplants/svr_model.joblib', 'rb') as f:
        
        model = joblib.load(f)
    y_pred = model.predict(x_test)
    test_mape_error = mean_absolute_percentage_error(y_pred,y_test)
    st.write('Accuracy %' ,round(100-(100*test_mape_error),2))
    #st.write('Next ',timeframe, ' Price' ,y_pred[-1])
    st.write('Stock Price Prediction Graph for Last 60 Weeks based on ML Model')
    y_test_df = pd.DataFrame(y_test, columns=['Closing Price'], index = index_date )
    y_pred_df = pd.DataFrame(y_pred, columns=['Closing Price'], index = index_date )
    plot_data(y_test_df,y_pred_df )
    #st.altair_chart(plot_data(y_test,y_pred , 'SVR'))

def predict_price(x_real_test, timeframe, test_index_date):
    if timeframe == 'Week':
        model_file = '/data/project12/svr_model.joblib'  
        index_date = test_index_date +  pd.to_timedelta(7, unit='D')
     
    else:
        model_file = '/data/project12/svr_model_month.joblib'
        index_date = test_index_date +  pd.to_timedelta(28, unit='D')
    with open(model_file, 'rb') as f:      
        model = joblib.load(f)
    y_pred_test = model.predict(x_real_test)
    
    df_predict = pd.DataFrame(y_pred_test, columns=['Predicted Closing Price'] )
    df_predict['Future Trading Date'] = index_date
    df_predict['Future Trading Date'] = pd.to_datetime(df_predict['Future Trading Date']).dt.round('D')
    df_predict_p = df_predict.copy()
    
    #df_predict['Future Trading Date'] = pd.to_datetime(df_predict['Future Trading Date'], format='%Y-%m-%d', errors='coerce')
    df_predict_p.index = pd.to_datetime(df_predict_p['Future Trading Date'], format='%y-%m-%d')
    df_predict_p.index.name = 'Future Trading Date'
    df_predict_p.drop(columns=['Future Trading Date'], inplace=True)
    st.write('Next ',timeframe, ' Price ' ,round(df_predict_p,1))
    return df_predict

    
    #c = alt.Chart(df_predict).mark_circle().encode(
    #x='Future Trading Date', y='Predicted Closing Price')
    #st.altair_chart(c, use_container_width=True)


    

# Plotting the predictions
def plot_data(Y_test,Y_hat):
    fig, ax = plt.subplots(1,1,figsize=(9,6))
    ax.plot(Y_test,c = 'r')
    ax.plot(Y_hat,c = 'y')
    ax.set(xlabel='Years', ylabel=' Close Price', title='Stock Prediction Graph')
    ax.grid()
    ax.legend(['Actual','Predicted'],loc = 'lower right')
    #st.pyplot(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    
def plot_animation(df):
    lines = alt.Chart(df).mark_line().encode(x=alt.X('date:T', axis=alt.Axis(title='date')),
                                             y=alt.Y('value:Q',axis=alt.Axis(title='value')),).properties(width=600,  height=300) 
    return lines

# This sidebar UI is a little search engine to select stocks and timeframe , And predict based on that.
def stock_select_predict( df):
    
    st.sidebar.markdown("# Stock Price Prediction Model")
    timeframes = ['Week', 'Month']

    # The user can pick which stock to slect for prediction.
    stock_codes = df['stock_code'].unique()
    stock = st.sidebar.selectbox("Select BSE500 Stock Ticker", stock_codes)
    stock_df = filter_input_data(stock,df )
    st.write('Stock Price Chart for Last 2 Decades')
    alt.Chart(st.line_chart(stock_df))  
    timeframe = st.sidebar.selectbox("Select Future TimeFrame", timeframes)  
    check_action = st.button("CHECK ACCURACY")    
    return stock_df, timeframe, check_action
    

    

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.  
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("StreamlitApp.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()
        st.caption("Disclaimer : The investments discussed or recommended in this stock price prediction model may not be suitable for all investors. Investors must make their own investment decisions based on their specific investment objectives and financial position and only after consulting such independent advisors as may be necessary.")


def run_the_app():
    input_path = "https://raw.githubusercontent.com/uttamk22/AIML/moneyplants/bse_stocks_technical_weekly_data_latest.csv"
    #input_path = "https://github.com/uttamk22/AIML/blob/moneyplants/bse_stocks_technical_weekly_data_latest.csv"
    df = create_input_data(input_path)
    stock_df, timeframe, check_action = stock_select_predict(df)
    window = 28
    if timeframe == 'Week':
        x_test, y_test, index_date, x_real_test, test_index_date = create_test_data(stock_df, window, 0)
    else:  
        x_test, y_test, index_date, x_real_test,test_index_date = create_test_data(stock_df, window, 3)
    
    if check_action:
        check_model(x_test, y_test, timeframe, index_date )
        
    predict_action = st.button("PREDICT THE FUTURE")  
    if predict_action:
        df_predict = predict_price(x_real_test, timeframe, test_index_date)
        #st.line_chart(df_predict)
    
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.plot(df_predict['Future Trading Date'],df_predict['Predicted Closing Price'], 'bs')
        ax.set(xlabel='Future Trading Date', ylabel='Predicted Close Price', title='Stock Prediction Model')
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        st.plotly_chart(fig)
        
        # lines = plot_animation(df)
        # line_plot = st.altair_chart(lines)
        # start_btn = st.button('Start')
        # N, burst, size = df.shape[0] , 1, 1
        # if start_btn:
        #     for i in range(1,N):
        #         step_df = df.iloc[0:size]
        #         lines = plot_animation(step_df)
        #         line_plot = line_plot.altair_chart(lines)
        #         size = i + burst
        #         if size >= N: 
        #             size = N - 1
        #         time.sleep(0.1)
        
    

if __name__ == "__main__":
    
    main()
