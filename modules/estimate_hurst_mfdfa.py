# -*- coding: utf-8 -*-
"""
Created on Thu May 16 2023

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import logging
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from MFDFA import MFDFA
from functools import partial
from multiprocessing import Pool, cpu_count


# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Estime Hurst exponent using Multifractal Detrended Fluctuation Analysis method ----
def estimate_hurst_mfdfa(
    y_data,
    start,
    end,
    points,
    q_orders,
    orders,
    log_path="../logs",
    log_filename="log_hurst_global",
    verbose=1
):
    """Hurst exponent using MDFA library
    Estimate Hurst exponent using MFDFA method:
        y_data: Data of time series to estimate Hurst exponent
        start: Start point in base 10 for lags in Multifractal Detrended Fluctuation Analysis (MF-DFA) method
        end: End  point in base 10 for lags in MF-DFA method
        points: Number of points in lags used for MF-DFA method
        q_orders: Exponent orders using in the fluctuation of MF-DFA method
        orders: Degrees of polynomials used to calculate detrended fluctuation analysis (DFA)
        log_path: Logs path
        log_filename: Log filename for output
        verbose: verbose
    """
    
    if isinstance(q_orders, int) == True:
        q_orders = [q_orders]
        
    if isinstance(orders, int) == 1:
        orders = [orders]
    
    try:
        # Select a band of lags from 10**start to 10**end ----
        lag = np.unique(np.logspace(start, end, points).astype(int))
        df_hurst = []
        
        # Select the power q ----
        for i in q_orders:
            # The order of the polynomial fitting (Detendred fluctuation analysis) ----
            for j in orders:
                lag, dfa = MFDFA(y_data, lag = lag, q = i, order = j)

                # And now we need to fit the line to find the slope. Don't
                # forget that since you are plotting in a double logarithmic
                # scales, you need to fit the logs of the results
                H_hat = np.polyfit(np.log(lag)[4:], np.log(dfa[4:]), 1)[0]

                # Final dataframe with regressions ----
                df_local_hurst = pd.DataFrame(
                    {
                        "q_order" : [i],
                        "dfa_degree" : [j],
                        "hurst" : H_hat
                    },
                    index = [0]
                )

                # Append to final dataframe ----
                df_hurst.append(df_local_hurst)
                
                # Function development ----
                if verbose >= 1:
                    with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                        file.write("Estimated Hurst {} for q = {} and dfa_degree = {}\n".format(H_hat[0], i, j))

    except Exception as e:
        df_hurst = []
        
        # Select the power q ----
        for i in q_orders:
            # The order of the polynomial fitting (Detendred fluctuation analysis) ----
            for j in orders:
                # Final dataframe with regressions ----
                df_local_hurst = pd.DataFrame(
                    {
                        "q_order" : [i],
                        "dfa_degree" : [j],
                        "hurst" : 0
                    },
                    index = [0]
                )
                
                # Append to final dataframe ----
                df_hurst.append(df_local_hurst)

                # Function development ----
                if verbose >= 1:
                    with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                        file.write("Non-estimated Hurst for q = {} and dfa_degree = {}\n".format(i, j))
                        file.write("{}\n".format(e))
    
    df_hurst = pd.concat(df_hurst).reset_index()
    del df_hurst["index"]

    return df_hurst

# Estime Hurst exponent over multiple Symbols ----
def estimate_hurst_mfdfa_df(
    df_data,
    points,
    n_step,
    log_path,
    log_filename,
    verbose,
    arg_list
):
    """Hurst exponent using MDFA library over different symbols
    Estimate Hurst exponent using MFDFA method:
        df_data: Dataframe with multiple symbols
        points: Number of points in lags used for MF-DFA method
        n_step: Number of skipped steps in original data (window size)
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        arg_list[0]: Exponent orders using in the fluctuation of MF-DFA method
        arg_list[1]: Degrees of polynomials used to calculate detrended fluctuation analysis (DFA)
    """
    
    # Definition of arg_list components ----
    q_orders = arg_list[0]
    orders = arg_list[1]
    
    df_hurst = []
    for i in df_data["symbol"].unique():
        df_aux = df_data[df_data["symbol"] == i]
        steps = df_aux[((df_aux["step"] % n_step == 0) | (df_aux["step"] == (df_aux.shape[0] - 1)))]["step"].unique()
        for j in steps:
            # Estimate initial and final lags ----
            end = np.log10(df_data[(df_data["symbol"] == i) & (df_data["step"] <= j)].shape[0]).astype(int)
            start = np.min([0.5, end])
        
            # Hurst exponent from log-return data ----
            df_local_1 = estimate_hurst_mfdfa(
                y_data = df_data[(df_data["symbol"] == i) & (df_data["step"] <= j)]["log_return"].values,
                start = start,
                end = end,
                points = points,
                q_orders = q_orders,
                orders = orders,
                log_path = log_path,
                log_filename = log_filename,
                verbose = verbose
            )
            df_local_1["symbol"] = i
            df_local_1["step"] = j
            df_local_1["time_series"] = "log-return"        

            # Hurst exponent from Absolute log-return data ----
            df_local_2 = estimate_hurst_mfdfa(
                y_data = df_data[(df_data["symbol"] == i) & (df_data["step"] <= j)]["absolute_log_return"].values,
                start = start,
                end = end,
                points = points,
                q_orders = q_orders,
                orders = orders,
                log_path = log_path,
                log_filename = log_filename,
                verbose = verbose
            )
            df_local_2["symbol"] = i
            df_local_2["step"] = j
            df_local_2["time_series"] = "absolute log-return"    

            # Hurst exponent from log-volatility data ----
            df_local_3 = estimate_hurst_mfdfa(
                y_data = df_data[(df_data["symbol"] == i) & (df_data["step"] <= j)]["log_volatility"].values,
                start = start,
                end = end,
                points = points,
                q_orders = q_orders,
                orders = orders,
                log_path = log_path,
                log_filename = log_filename,
                verbose = verbose
            )
            df_local_3["symbol"] = i
            df_local_3["step"] = j
            df_local_3["time_series"] = "log-return volatility"    

            # Append to final dataframe ----
            df_hurst.append(df_local_1)
            df_hurst.append(df_local_2)
            df_hurst.append(df_local_3)

            # Function development ----
            if verbose >= 1:
                with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                    file.write("Estimated Hurst exponent for Symbol {}, Step {} and length {}\n".format(i, j, np.exp(end)))
                
    df_hurst = pd.concat(df_hurst).reset_index()
    del df_hurst["index"]

    return df_hurst

# Deployment of parallel run in function of arguments list ----
def parallel_run(
    fun,
    arg_list,
    tqdm_bar=False
):
    """Parallel run
    Implement parallel run in arbitrary function with input arg_list:
        fun: Function to implement in parallel
        arg_list: List of arguments to pass in function
        tqdm_bar: Progress bar flag
    """
    
    if tqdm_bar:
        m = []
        with Pool(processes = cpu_count()) as p:
            with tqdm(total = len(arg_list), ncols = 60) as pbar:
                for _ in p.imap(fun, arg_list):
                    m.append(_)
                    pbar.update()
            p.terminate()
            p.join()
    else:
        p = Pool(processes = cpu_count())
        m = p.map(fun, arg_list)
        p.terminate()
        p.join() 
    return m

# Estimate mean and variance parameters ----
def estimate_hurst_mfdfa_global(
    df_data,
    minimal_steps=60,
    points=100,
    n_step=15,
    q_orders=2,
    orders=1,
    log_path="../logs",
    log_filename="log_hurst_global",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of Hurst on multiple symbols using MFDFA library and parallel run
    Estimation of Hurst exponent in parallel loop:
        df_data: Dataframe with multiple symbols
        minimal_steps: Minimum points used for regression of temporal fluctuation scaling (TFS)
        points: Number of points in lags used for MF-DFA method
        n_step: Number of skipped steps in original data (window size)
        q_orders: Exponent orders using in the fluctuation of MF-DFA method
        orders: Degrees of polynomials used to calculate detrended fluctuation analysis (DFA)
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        tqdm_bar: Progress bar flag
    """
    
    # Auxiliary function for estimation of mean and variance parameters ----
    fun_local = partial(
        estimate_hurst_mfdfa_df,
        df_data,
        points,
        n_step,
        log_path,
        log_filename,
        verbose
    )
    
    # Definition of arg_list sampling ----
    arg_list = np.array(np.meshgrid(q_orders, orders)).T.reshape(-1, 2).tolist()
    
    # Parallel loop for mean and variance parameters estimation ----
    df_fts_parameters = parallel_run(fun = fun_local, arg_list = arg_list, tqdm_bar = tqdm_bar)
    df_fts_parameters = pd.concat(df_fts_parameters)
    df_fts_parameters = df_fts_parameters[df_fts_parameters["step"] != 0]
    
    return df_fts_parameters

