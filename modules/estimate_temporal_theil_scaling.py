# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 2023

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import logging
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Temporal fluctuation scaling (TFS) ----
def temporal_theil_scaling(mean, coefficient_tts, exponent_tts):
    """Estimation of temporal Theil scaling (TTS)
    Estimation of temporal Theil scaling:
        mean: Mean of a sample in time t
        coefficient_tts: Coefficient for estimation of TTS as power law
        exponent tts: Exponent for estimation of TTS as power law
    """    
    return coefficient_tts * np.power(np.abs(1 - mean / np.max(mean)), exponent_tts)

# Estimation of p-norm ----
def estimate_p_norm(x, y, p):
    if p == 0:
        z = np.exp(0.5 * np.mean(np.log(np.power(np.abs(x-y), 2))))
    else:
        z = np.power(np.abs(x - y), 1 / p)
    return np.mean(z)

# Estimation of coefficient of determination R2 ----
def estimate_coefficient_of_determination(y, y_fitted):
    return 1 - np.sum(np.power(y - y_fitted, 2)) / np.sum(np.power(y - np.mean(y), 2))

# Estimate mean and shannon index parameters ----
def estimate_tts_parameters_local(
    df_fts,
    p_norm,
    log_path,
    log_filename,
    verbose,
    arg_list
):
    """Estimation of parameters local
    Estimation of parameters of temporal Theil scaling:
        df_fts: Dataframe with multiple financial time series after applied diffusive algorithm
        p_norm: p-norm selection
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        arg_list[0]: Symbol to filter in time series
        arg_list[1]: Number of steps used to filter data
    """
    
    # Definition of arg_list components ----
    symbol = arg_list[0]
    n_step = arg_list[1]
    
    try:
        # Filtration of information ----
        df_parameters = pd.DataFrame()
        df_fts = df_fts[((df_fts["time"] <= n_step) & (df_fts["symbol"] == symbol))]
        for j in sorted(df_fts["time_series"].unique().tolist()):

            # Estimation of parameters (TTS) ----
            df_aux = df_fts[df_fts["time_series"] == j]
            df_aux["shannon_index"] = df_aux["shannon_index"] / df_aux["shannon_index"].max()
            df_aux["mean_value"] = 1 - df_aux["mean_value"] / df_aux["mean_value"].max()
            popt, pcov = curve_fit(temporal_theil_scaling, df_aux["mean_value"], df_aux["shannon_index"])

            # Estimation of uncertainty in estimate parameters (TTS) ----
            ee = np.sqrt(np.diag(pcov))

            # Estimation of value with estimated parameters (TTS) ----
            estimated = temporal_theil_scaling(df_aux["mean_value"], *popt).values

            # Estimation of average error with residuals (TTS) ----
            ae = estimate_p_norm(x = df_aux["shannon_index"].values, y = estimated, p = p_norm)

            # Estimation of R squared (TTS) ----
            r2 = estimate_coefficient_of_determination(y = df_aux["shannon_index"].values, y_fitted = estimated)

            # Final dataframe with regressions ----
            df_parameters = df_parameters._append(
                pd.DataFrame(
                    {
                        "symbol" : [symbol],
                        "max_step" : [n_step],
                        "time_series" : [j],
                        "p_norm" : [p_norm],
                        "coefficient_tts" : [popt[0]],
                        "error_coefficient_tts" : [ee[0]],
                        "exponent_tts" : [popt[1]],
                        "error_exponent_tts" : [ee[1]],
                        "average_error_tts" : [ae],
                        "rsquared_tts" : [r2]
                    },
                    index = [0]
                )
            )
        
        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Estimated TTS parameters for {} with {} max steps and {}-norm\n".format(symbol, n_step, p_norm))

    except Exception as e:
        # Final dataframe with regressions ----
        df_parameters = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "max_step" : [n_step, n_step, n_step],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "p_norm" : [p_norm, p_norm, p_norm],
                "coefficient_tts" : [0, 0, 0],
                "error_coefficient_tts" : [0, 0, 0],
                "exponent_tts" : [0, 0, 0],
                "error_exponent_tts" : [0, 0, 0],
                "average_error_tts" : [0, 0, 0],
                "rsquared_tts" : [0, 0, 0]
            },
            index = [0, 1, 2]
        )
        
        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("No estimated TTS parameters for {} with {} max steps and {}-norm\n".format(symbol, n_step, p_norm))
                file.write("{}\n".format(e))
        
    return df_parameters

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

# Estimate mean and shannon index parameters ----
def estimate_tts_parameters(
    df_fts,
    minimal_steps=30,
    p_norm=2,
    log_path="../logs",
    log_filename="log_tts_evolution",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of parameters global
    Estimation of parameters of temporal Theil scaling in parallel loop:
        df_fts: Dataframe with multiple financial time series
        minimal_steps: Minimum points used for regression of temporal Theil scaling (TTS)
        p_norm: p-norm selection
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        tqdm_bar: Progress bar flag
    """
    
    # Auxiliary function for estimation of mean and variance parameters ----
    fun_local = partial(
        estimate_tts_parameters_local,
        df_fts,
        p_norm,
        log_path,
        log_filename,
        verbose
    )
    
    # Definition of arg_list sampling ----
    arg_list = df_fts[df_fts["time"] >= minimal_steps][["symbol", "time"]].drop_duplicates().values.tolist()
    
    # Parallel loop for mean and variance parameters estimation ----
    df_fts_parameters = parallel_run(fun = fun_local, arg_list = arg_list, tqdm_bar = tqdm_bar)
    df_fts_parameters = pd.concat(df_fts_parameters).reset_index()
    df_fts_parameters = df_fts_parameters[df_fts_parameters["max_step"] != 0]
    del [df_fts_parameters["index"]]
    
    return df_fts_parameters
