# -*- coding: utf-8 -*-
"""
This is the main file to get all major financial ratio and rolling ratio and other important graph
"""
import os
import mod_financial_ratio as r
import mod_rolling as m
import mod_input_output as i
import mod_plot as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from pylab import rcParams
rcParams['figure.figsize'] = 18, 12

if __name__ == '__main__':
    ### Change working directory
    # Get working directory
    os.getcwd()
    # Set working director
    data_file_path = r'C:\Users\ZFang\Desktop\TeamCo\Financial Ratio\\'
    os.chdir(data_file_path)
    ### Define constant variable
    # The number of month at the beginning that has no available data
    # Since start from 2007, the first six months doesn't have performance data, 
    # I set the starting gap is 6 months
    start_gap = 6
    index_name = ('TeamCo Client Composite', 'HFRI Fund Weighted Composite Index', 'HFRI Fund of Funds Composite Index')
    columns_name = ['TeamCo Client Composite', 'HFRI Fund Weighted Composite Index', 'HFRI Fund of Funds Composite Index', 'Russell 3000']
    ### Plotting stype
    
    ### Read the file
    df_data = i.concat_data('Fund Analysis.xlsx')
    ### Calculate Annulized Return
    Annulized_Return_df = r.annulized_return(df_data, index_name, start_gap=start_gap)
    ### Calculate Calendar Return
    Calendar_Return_df = r.calendar_return(df_data, index_name, 2007, 2015, 1)
    ### Calculate Downside Deviation, given order of two
    Downside_Deviation_df = r.downside_std(df_data, index_name, threshold=0, order=2, start_gap=start_gap)
    ### Calculate Sortino ratio
    Sortino_df = r.sortino_ratio(df_data, index_name, MAR=0, threshold=0, order=2, start_gap=start_gap)
    ### Calculate Sharp ratio
    Sharpe_df=r.sharpe_ratio(df_data, index_name, benchmark=0.02, start_gap=start_gap)
    ### Standard Deviation
    Standard_deviation_df = r.standard_deviation(df_data, index_name, start_gap=start_gap)
    ### Beta matrix
    Beta_df = r.beta_table(df_data, index_name, market_index='Russell 3000', start_gap=start_gap, condition = None)
    ### Positive Beta matrix
    Beta_df_p = r.beta_table(df_data, index_name, market_index='Russell 3000', start_gap=start_gap, condition = 'Positive')
    ### Non Negative Beta matrix
    Beta_df_np = r.beta_table(df_data, index_name, market_index='Russell 3000', start_gap=start_gap, condition = 'Non-positive')
    ### Omega Ratio
    Omega_df = r.omega_ratio(df_data, index_name, MAR=0, threshold=0, order=1, start_gap=start_gap)
    ### Correlation table
    Corr_df = r.corr_table(df_data, index_name, target_mkt_index='Russell 3000', start_gap=start_gap, condition=None)
    ### Positive Correlation table
    Corr_df_p = r.corr_table(df_data, index_name, target_mkt_index='Russell 3000', start_gap=start_gap, condition='Positive')
    ### Positive Correlation table
    Corr_df_np = r.corr_table(df_data, index_name, target_mkt_index='Russell 3000', start_gap=start_gap, condition='Non-positive')    
    ### Summary table
    Summary_table_df = r.summary_table(df_data,index_name, 
        columns=['Batting Average', 'Omega Ratio', 'Up Months', 'Down Months', 'Slugging Ratio', 'Up-Capture Russell', 'Down-Capture Russell'], 
        market_index='Russell 3000',MAR=0, threshold=0, order=1, start_gap=start_gap)
    ### Daily maximum Drawdown for differnt portfolio
    max_dd_df =m.time_drawdown(df_data, 'TeamCo Client Composite', start_gap=start_gap)
    ### Maximum Drawdown for given time window
    max_dd = m.max_drawdown(df_data, 'TeamCo Client Composite', start_gap=start_gap)
    
    
    ### Rolling beta
    rolling_beta_df = m.rolling_beta(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6)
    ### Rolling annulized return
    rolling_annual_return_df = m.rolling_annulized_return(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6)
    ### Cummulative return
    cum_return_df = m.cumulative_return(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6)
    ### Rolling sortino ratio
    rolling_sortino_ratio_df = m.rolling_sortino_ratio(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6, MAR=0, threshold=0, order=2)
    ### Rolling omega ratio
    rolling_omega_ratio_df = m.rolling_omega_ratio(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6, MAR=0)
    ### Rolling sharp ratio
    rolling_sharpe_ratio_df = m.rolling_sharpe_ratio(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6, benchmark=0.02)
    ### Rolling alpha
    rolling_alpha_df = m.rolling_alpha(df_data, 
        columns_name=columns_name, 
        window_length=36, min_periods=36, start_gap=6)
    ### Rolling correlation
    rolling_corr_df = m.rolling_corr(df_data, 
        columns_name=columns_name, 
        
        target_mkt_index='Russell 3000', window_length=36, min_periods=36, start_gap=6)
    ### Draw Down
    dd_df = 100* m.draw_down(df_data, columns_name, start_gap)
    
    
    ### Generate graph and save them to the pdf file
    p.graph_gen('Rolling Ratio Figure and Radar Chart Result.pdf', index_name, rolling_annual_return_df, cum_return_df, \
                rolling_alpha_df,rolling_beta_df, rolling_corr_df, rolling_sharpe_ratio_df, \
                rolling_sortino_ratio_df,rolling_omega_ratio_df, dd_df, Beta_df, Beta_df_p, \
                Beta_df_np, Corr_df, Corr_df_p, Corr_df_np)

    
    ### Output all static dataframe into excel file
    dfs = [Annulized_Return_df,Calendar_Return_df,Sharpe_df,Sortino_df,\
           Standard_deviation_df,Downside_Deviation_df,Beta_df,Beta_df_p,\
           Beta_df_np,Omega_df,Corr_df,Corr_df_p,Corr_df_np,Summary_table_df]
    i.multiple_dfs(dfs, 'Financial Ratio', 'Financial Ratio Result.xlsx', 1)
    
    ### Output all rolling data to seperated sheet in excel file
    rolling_df_list = [rolling_beta_df,rolling_annual_return_df,cum_return_df,\
                       rolling_sortino_ratio_df,rolling_omega_ratio_df,rolling_sharpe_ratio_df,\
                       rolling_alpha_df,rolling_corr_df]
    i.multiple_sheets(rolling_df_list, 'Rolling Result.xlsx')
