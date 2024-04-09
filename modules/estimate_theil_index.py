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
from functools import partial
from multiprocessing import Pool, cpu_count


# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Generate diffusive paths (diffusive trajectory time series (DTTS)) through diffusive algorithm ----
def estimate_diffusive_algorithm(
    df_data,
    s_vector,
    log_path="../logs",
    log_filename="log_shannon_index",
    verbose=1
):
    """Diffusive algorithm
    Estimate diffusive algorithm:
        df_data: Dataframe of time series to estimate diffusive algorithm
        s_vector: S values used for diffusive algorithm lags
        log_path: Logs path
        log_filename: Log filename for output
        verbose: verbose
    """
    
    if isinstance(s_vector, int) == True:
        s_vector = [s_vector]
    
    try:
        df_dtts = pd.DataFrame()
        max_index = df_data["step"].max()

        for i in s_vector:
            # Construction of diffusive path as a realization of accumulated value over the time ----
            df_aux = df_data[df_data["step"] >= i]
            df_dtts = df_dtts._append(
                pd.DataFrame(
                    {
                        "time": df_aux["step"].values,
                        "sub_time": np.repeat(i, df_aux.shape[0]).tolist(),
                        "diffusive_log_return": df_aux["log_return"].cumsum().values,
                        "diffusive_absolute_log_return": df_aux["absolute_log_return"].cumsum().values,
                        "diffusive_log_volatility": df_aux["log_volatility"].cumsum().values,
                    }
                ),
                ignore_index = True
            )
            
            # Function development ----
            if verbose >= 1:
                with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                    file.write("Estimated diffusive path sub-time {}\n".format(i))

    except Exception as e:
        df_dtts = pd.DataFrame(
            {
                "time": [-1],
                "sub_time": [-1],
                "diffusive_log_return": [0],
                "diffusive_absolute_log_return": [0],
                "diffusive_log_volatility": [0]
            }
        )

        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Non-estimated diffusive trajectory\n")
                file.write("{}\n".format(e))
    
    df_dtts.sort_values(by = ["time", "sub_time"], inplace = True)
    df_dtts = df_dtts.reset_index()
    return df_dtts

# Estime Shannon index ----
def estimate_shannon_index(
    y_data,
    log_path="../logs",
    log_filename="log_shannon_index",
    verbose=1
):
    """Theil index
    Estimate Shannon index from Theil index:
        y_data: Data of time series to estimate Shannon index
        log_path: Logs path
        log_filename: Log filename for output
        verbose: verbose
    """
    
    try:
        mean_value = np.nanmean(y_data)
        y_data_norm = y_data / mean_value
        theil_index = (1 / y_data_norm.shape[0]) * np.nansum(y_data_norm * np.log(y_data_norm))

        # Final dataframe with parameters ----
        df_shannon_index = pd.DataFrame(
            {
                "length" : [y_data_norm.shape[0]],
                "mean_value" : [mean_value],
                "theil_index" : [theil_index],
                "shannon_index" : np.log(y_data_norm.shape[0]) - theil_index
            },
            index = [0]
        )

        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Estimated Theil index {}\n".format(theil_index))

    except Exception as e:
        df_shannon_index = pd.DataFrame(
            {
                "length" : [-1],
                "mean_value" : [0],
                "theil_index" : [0],
                "shannon_index" : [0]
            },
            index = [0]
        )

        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Non-estimated Theil index\n")
                file.write("{}\n".format(e))

    return df_shannon_index

# Estime diffusive paths (diffusive trajectory time series (DTTS)) over one Symbol ----
def estimate_diffusive_algorithm_df(
    df_data,
    log_path,
    log_filename,
    verbose,
    arg_list
):
    """Diffusive algorithm over different symbols
    Estimate diffusive algorithm over different symbols:
        df_data: Dataframe of time series to estimate diffusive algorithm with multiple symbols
        log_path: Logs path
        log_filename: Log filename for output
        verbose: verbose
        arg_list[0]: S values used for diffusive algorithm lags
        arg_list[1]: Symbol or ticker of financial time series
    """
    
    # Definition of arg_list components ----
    s_vector = arg_list[0]
    symbol = arg_list[1]
    
    if isinstance(s_vector, int) == True:
        s_vector = [s_vector]
    
    # Function development ----
    if verbose >= 1:
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write("----------------------------------- {} -----------------------------------\n".format(symbol))

    # Local estimation of Diffusive Algortihm ----
    df_dtts = estimate_diffusive_algorithm(
        df_data = df_data[df_data["symbol"] == symbol],
        s_vector = s_vector,
        log_path = log_path,
        log_filename = log_filename,
        verbose = verbose
    )
    df_dtts["symbol"] = symbol
    column_1 = df_dtts.pop("symbol")
    df_dtts.insert(0, "symbol", column_1)
    
    return df_dtts

# Estime Shannon index over one Symbols after construct DTTS ----
def estimate_shannon_index_df(
    df_dtts_data,
    log_path,
    log_filename,
    verbose,
    arg_list
):
    """Shannon index over different symbols
    Estimate temporal Theil scaling data:
        df_dtts_data: Dataframe with diffusive algorithm and multiple symbols
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        arg_list[0]: time values used for clustering data
        arg_list[1]: Symbol or ticker of financial time series
    """
    
    # Definition of arg_list components ----
    time_vector = arg_list[0]
    symbol = arg_list[1]
    
    if isinstance(time_vector, int) == True:
        time_vector = [time_vector]
    
    df_shannon_index = df_dtts_data[((df_dtts_data["symbol"] == symbol) & (df_dtts_data["time"].isin(time_vector)))]

    try:
        # Mean value log-return, absolute log return and volatility of log return data ----
        mean_value_lr = np.nanmean(df_shannon_index["diffusive_log_return"].values)
        mean_value_la = np.nanmean(df_shannon_index["diffusive_absolute_log_return"].values)
        mean_value_lv = np.nanmean(df_shannon_index["diffusive_log_volatility"].values)
        
        # Normalized log-return, absolute log return and volatility of log return data ----
        diffusive_lr_norm = df_shannon_index["diffusive_log_return"] / mean_value_lr
        diffusive_la_norm = df_shannon_index["diffusive_absolute_log_return"] / mean_value_la
        diffusive_lv_norm = df_shannon_index["diffusive_log_volatility"] / mean_value_lv
        
        # Shannon index log-return, absolute log return and volatility of log return data ----
        theil_lr = (1 / df_shannon_index.shape[0]) * np.nansum(diffusive_lr_norm * np.log(diffusive_lr_norm))
        theil_la = (1 / df_shannon_index.shape[0]) * np.nansum(diffusive_la_norm * np.log(diffusive_la_norm))
        theil_lv = (1 / df_shannon_index.shape[0]) * np.nansum(diffusive_lv_norm * np.log(diffusive_lv_norm))
        
        shannon_index_lr = np.log(df_shannon_index.shape[0]) - theil_lr
        shannon_index_la = np.log(df_shannon_index.shape[0]) - theil_la
        shannon_index_lv = np.log(df_shannon_index.shape[0]) - theil_lv
        
        # Final dataframe with parameters ----
        df_shannon_index = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "time" : [time_vector[0], time_vector[0], time_vector[0]],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "mean_value" : [mean_value_lr, mean_value_la, mean_value_lv],
                "theil_index" : [theil_lr, theil_la, theil_lv],
                "shannon_index" : [shannon_index_lr, shannon_index_la, shannon_index_lv]
            },
            index = [0, 1, 2]
        )

        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Estimated Theil index for {} and time {}\n".format(symbol, time_vector[0]))

    except Exception as e:
        df_shannon_index = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "time" : [-1, -1, -1],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "mean_value" : [0, 0, 0],
                "theil_index" : [0, 0, 0],
                "shannon_index" : [0, 0, 0]
            },
            index = [0, 1, 2]
        )

        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Non-estimated Theil index for {} and time {}\n".format(symbol, time_vector[0]))
                file.write("{}\n".format(e))

    return df_shannon_index

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

# Estimate DTTS in parallel loop ----
def estimate_diffusive_algorithm_global(
    df_data,
    minimal_steps=0,
    log_path="../logs",
    log_filename="log_dtts_global",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of parameters global
    Estimation of parameters of temporal Theil scaling in parallel loop:
        df_fts: Dataframe with multiple financial time series
        minimal_steps: Minimum points used for subtime of DTTS
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        tqdm_bar: Progress bar flag
    """
    
    # Auxiliary function for estimation of mean and variance parameters ----
    fun_local = partial(
        estimate_diffusive_algorithm_df,
        df_data,
        log_path,
        log_filename,
        verbose
    )
    
    # Definition of arg_list sampling ----
    arg_list = df_data[df_data["step"] >= minimal_steps][["step", "symbol"]].drop_duplicates().values.tolist()
    
    # Parallel loop for mean and variance parameters estimation ----
    df_dtts_parameters = parallel_run(fun = fun_local, arg_list = arg_list, tqdm_bar = tqdm_bar)
    df_dtts_parameters = pd.concat(df_dtts_parameters).reset_index()
    df_dtts_parameters = df_dtts_parameters[df_dtts_parameters["time"] >= 0]
    del [df_dtts_parameters["index"]]
    
    return df_dtts_parameters

# Estimate Shannon index in parallel loop ----
def estimate_shannon_index_global(
    df_dtts_data,
    minimal_steps=0,
    log_path="../logs",
    log_filename="log_theil_global",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of parameters global
    Estimation of parameters of temporal Theil scaling in parallel loop:
        df_fts: Dataframe with multiple financial time series
        minimal_steps: Minimum points used for time of DTTS
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        tqdm_bar: Progress bar flag
    """
    
    # Auxiliary function for estimation of mean and variance parameters ----
    fun_local = partial(
        estimate_shannon_index_df,
        df_dtts_data,
        log_path,
        log_filename,
        verbose
    )
    
    # Definition of arg_list sampling ----
    arg_list = df_dtts_data[df_dtts_data["time"] >= minimal_steps][["time", "symbol"]].drop_duplicates().values.tolist()
    
    # Parallel loop for mean and variance parameters estimation ----
    df_tts_parameters = parallel_run(fun = fun_local, arg_list = arg_list, tqdm_bar = tqdm_bar)
    df_tts_parameters = pd.concat(df_tts_parameters).reset_index()
    df_tts_parameters = df_tts_parameters[df_tts_parameters["time"] >= 0]
    del [df_tts_parameters["index"]]
    
    return df_tts_parameters

