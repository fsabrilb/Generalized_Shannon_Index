# -*- coding: utf-8 -*-
"""
Created on Thu May 23 2023

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import re
import sys
import warnings
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

from scipy.optimize import curve_fit
from scipy.stats import percentileofscore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "olive", "gray", "cyan"]

# Estimate tfs parameters ----
def prepare_data(
    df_fts,
    df_hurst,
    df_tfs,
    df_tts,
    interval,
    threshold_n_data_hurst=2048,
    threshold_tfs=0,
    threshold_tts=0  
):
    """Preparation of data for plotting
    Join original data with optimal window size data:
        df_fts: Dataframe with multiple financial time series
        df_hurst: Dataframe with Hurst exponent for different q-values
        df_tfs: Dataframe with temporal fluctuation scaling parameters
        df_tts: Dataframe with temporal Theil scaling parameters
        interval: Select transformation for estimation of time between minimum and maximum date
        threshold_n_data_hurst: Threshold for filtering the number of minimal length of time series to estimate Hurst exponent
        threshold_tfs: Threshold of determination coefficient of temporal fluctuation scaling (TFS)
        threshold_tfs: Threshold of determination coefficient of temporal Theil scaling (TTS)
    """
    
    # Estimation of interval of time for each ticker ----
    df_fts["min_date"] = pd.to_datetime(df_fts.groupby(["symbol"])["date"].transform("min"), errors = "coerce", infer_datetime_format = True)
    df_fts["max_date"] = pd.to_datetime(df_fts.groupby(["symbol"])["date"].transform("max"), errors = "coerce", infer_datetime_format = True)
    
    interval_dict = {"years" : "Y", "months" : "M", "weeks" : "W", "days" : "D", "hours" : "h", "minutes" : "m", "seconds" : "s", "milliseconds" : "ms"}
    df_fts["duration"] = (df_fts["max_date"] - df_fts["min_date"]) / np.timedelta64(1, interval_dict[interval])
    df_dates = df_fts[["symbol", "min_date", "max_date", "duration"]].drop_duplicates(["symbol", "min_date", "max_date", "duration"])
    
    # Log-return data ----
    df_logr = (
        df_fts[["date", "symbol", "step", "cummean_log_return", "cumvariance_log_return"]]
            .rename(columns = {"cummean_log_return" : "cummean", "cumvariance_log_return" : "cumvariance"})
    )
    df_logr["time_series"] = "log-return"
    
    # Absolute log-return data ----
    df_loga = (
        df_fts[["date", "symbol", "step", "cummean_absolute_log_return", "cumvariance_absolute_log_return"]]
            .rename(columns = {"cummean_absolute_log_return" : "cummean", "cumvariance_absolute_log_return" : "cumvariance"})
    )
    df_loga["time_series"] = "absolute log-return"
    
    # Log-return volatility data ----
    df_logv = (
        df_fts[["date", "symbol", "step", "cummean_log_volatility", "cumvariance_log_volatility"]]
            .rename(columns = {"cummean_log_volatility" : "cummean", "cumvariance_log_volatility" : "cumvariance"})
    )
    df_logv["time_series"] = "log-return volatility"
    
    # Merge final data (Hurst exponent and Temporal Fluctuation Scaling (TFS)) ----
    df_plot_data = (
        pd.concat([df_logr, df_loga, df_logv])
            .merge(df_dates, left_on = ["symbol"], right_on = ["symbol"])
            .merge(df_hurst, left_on = ["symbol", "time_series", "step"], right_on = ["symbol", "time_series", "step"])
            .merge(df_tfs, left_on = ["symbol", "time_series", "step"], right_on = ["symbol", "time_series", "max_step"])
            .merge(df_tts, left_on = ["symbol", "time_series", "step"], right_on = ["symbol", "time_series", "max_step"])
    )
    
    df_plot_data["coefficient_tfs"].replace([np.inf, -np.inf], np.nan, inplace = True)
    df_plot_data["coefficient_tts"].replace([np.inf, -np.inf], np.nan, inplace = True)
    df_plot_data = df_plot_data[
        (
            (df_plot_data["coefficient_tfs"].notna()) &
            (df_plot_data["hurst"] != 0) &
            (df_plot_data["exponent_tfs"] != 0) &
            (df_plot_data["exponent_tts"] != 0)
        )
    ]    
    
    df_plot_data = df_plot_data.rename(columns = {"p_norm_x" : "p_norm_tfs", "p_norm_y" : "p_norm_tts"})
    df_plot_data = df_plot_data[
        (
            ((df_plot_data["rsquared_tfs"] >= threshold_tfs) | (df_plot_data["rsquared_tts"] >= threshold_tts)) &
            (df_plot_data["step"] >= threshold_n_data_hurst)
        )
    ]
    del [df_plot_data["max_step_x"], df_plot_data["max_step_y"]]
        
    return df_plot_data

# Estimation of coefficient of determination R2 ----
def estimate_coefficient_of_determination(y, y_fitted):
    return 1 - np.sum(np.power(y - y_fitted, 2)) / np.sum(np.power(y - np.mean(y), 2))

# Estimation of p-norm ----
def estimate_p_norm(x, y, p):
    if p == 0:
        z = np.exp(0.5 * np.mean(np.log(np.power(np.abs(x-y), 2))))
    else:
        z = np.power(np.abs(x - y), 1 / p)
    return np.mean(z)

# Plot evolution of Hurst, Temporal Fluctuation Scaling (TFS) and Temporal Theil Scaling (TTS) parameters ----
def plot_hurst_tfs_tts_evolution(
    df_fts,
    df_hurst,
    df_tfs,
    df_tts,
    interval,
    symbols,
    q_values,
    degree_tfs,
    degree_tts,
    width,
    height,
    threshold_n_data_hurst=2048,
    threshold_tfs=0,
    threshold_tts=0,
    markersize=2,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    usetex=False,
    n_cols=4,
    n_x_breaks=10,
    n_y_breaks=10,
    fancy_legend=True,
    dpi=150,
    save_figures=True,
    output_path="../output_files",
    information_name="",
    input_generation_date="2023-05-23"
):
    """Preparation of data for plotting
    Join original data with optimal window size data:
        df_fts: Dataframe with multiple financial time series
        df_hurst: Dataframe with optimal window size per financial time series
        df_tfs: Dataframe with temporal fluctuation scaling parameters
        df_tts: Dataframe with temporal Theil scaling parameters
        interval: Select transformation for estimation of time between minimum and maximum date
        symbols: Symbols of the financial time series plotted
        q_values: q values used to filter Generalized hurst values
        degree_tfs: Degree of polynomial fitting used for Hurst exponent as function of TFS exponent
        degree_tts: Degree of polynomial fitting used for Hurst exponent as function of TTS exponent
        width: Width of final plot
        height: Height of final plot
        threshold_n_data_hurst: Threshold for filtering the number of minimal length of time series to estimate Hurst exponent
        threshold_tfs: Threshold of determination coefficient of temporal fluctuation scaling (TFS)
        threshold_tfs: Threshold of determination coefficient of temporal Theil scaling (TTS)
        markersize: Marker size as in plt.plot()
        fontsize_labels: Font size in axis labels
        fontsize_legend: Font size in legend
        usetex: Use LaTeX for renderized plots
        n_cols: Number of columns in legend
        n_x_breaks: Number of divisions in x-axis
        n_y_breaks: Number of divisions in y-axis
        fancy_legend: Fancy legend output
        dpi: Dot per inch for output plot
        save_figures: Save figures flag
        output_path: Output path where figures is saved
        information_name: Name of the output plot
        input_generation_date: Date of generation (control version)
    """
    
    # Plot data and define loop over symbols ----
    df_fts = df_fts[df_fts["symbol"].isin(symbols)]
    df_hurst = df_hurst[df_hurst["q_order"].isin(q_values)]
    df_tfs = df_tfs[df_tfs["symbol"].isin(symbols)]
    df_tts = df_tts[df_tts["symbol"].isin(symbols)]
    
    df_graph = prepare_data(
        df_fts = df_fts,
        df_hurst = df_hurst,
        df_tfs = df_tfs,
        df_tts = df_tts,
        interval = interval,
        threshold_n_data_hurst = threshold_n_data_hurst,
        threshold_tfs = threshold_tfs,
        threshold_tts = threshold_tts
    )
    
    df_graph = df_graph[df_graph["time_series"] != "log-return"]
    loop_index = sorted(df_graph["symbol"].unique().tolist())
    
    # Begin plot inputs ----
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": usetex,
            "pgf.rcfonts": False
        }
    )
    
    fig1, ax1 = plt.subplots(len(loop_index), 2) # TFS exponent Evolution
    fig2, ax2 = plt.subplots(len(loop_index), 2) # TTS exponent Evolution
    fig3, ax3 = plt.subplots(len(loop_index), 2) # Hurst exponent Evolution
    fig4, ax4 = plt.subplots(len(loop_index), 2) # Hurst exponent vs TFS exponent
    fig5, ax5 = plt.subplots(len(loop_index), 2) # Hurst exponent vs TTS exponent
    fig1.set_size_inches(w = width, h = height)
    fig2.set_size_inches(w = width, h = height)
    fig3.set_size_inches(w = width, h = height)
    fig4.set_size_inches(w = width, h = height)
    fig5.set_size_inches(w = width, h = height)
    counter = 0
    
    df_hurst_tfs = pd.DataFrame()
    df_hurst_tts = pd.DataFrame()
    
    for i in loop_index:
        counter_i = 0
        for j in sorted(df_graph[df_graph["symbol"] == i]["time_series"].unique().tolist()):
            # Filter information ----
            df_aux = df_graph[((df_graph["symbol"] == i) & (df_graph["time_series"] == j))]
        
            # Parameters (Dates, Mean Average Error (MAE) and Rsquared percentile (R_percentile))----
            dates_j = pd.to_datetime(df_aux["date"].unique(), errors = "coerce")
            time_labels = pd.date_range(start = dates_j.min(), end = dates_j.max(), periods = n_x_breaks).strftime("%Y-%m-%d")
            ave_tfs_j = df_aux["average_error_tfs"]
            ave_tts_j = df_aux["average_error_tts"]
            rsquared_tfs_j = round(percentileofscore(df_aux["rsquared_tfs"], threshold_tfs), 2)
            rsquared_tts_j = round(percentileofscore(df_aux["rsquared_tts"], threshold_tts), 2)
            
            # Extract empirical data (All dates, exponents information and others) ----
            dates_tfs_j = pd.to_datetime(df_aux["date"], errors = "coerce")
            dates_tts_j = pd.to_datetime(df_aux["date"], errors = "coerce")
            exponent_tfs_j = df_aux["exponent_tfs"]
            exponent_tts_j = df_aux["exponent_tts"]
            coefficient_tfs_j = df_aux["coefficient_tfs"]
            coefficient_tts_j = df_aux["coefficient_tts"]
            
            error_exponent_tfs_j = df_aux["error_exponent_tfs"]
            error_exponent_tts_j = df_aux["error_exponent_tts"]
            error_coefficient_tfs_j = df_aux["error_coefficient_tfs"]
            error_coefficient_tts_j = df_aux["error_coefficient_tts"]
            
            # Plot graphs ----
            if len(loop_index) == 1:
                # Plot graph (Temporal Fluctuation Scaling) ----
                plot_1 = ax1[counter_i].plot(
                    dates_tfs_j,
                    exponent_tfs_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                #ax1[counter_i].fill_between(
                #    dates_tfs_j,
                #    exponent_tfs_j - error_exponent_tfs_j,
                #    exponent_tfs_j + error_exponent_tfs_j,
                #    alpha = 0.19,
                #    facecolor = colors[counter_i],
                #    interpolate = True
                #)
                ax1[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax1[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax1[counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax1[counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax1[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax1[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax1[counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                #ax1[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax1[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax1[counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax1[counter_i].set_ylabel("TFS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax1[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax1[counter_i].set_title(
                    r"({}) $MAE_p={}$, $q(R^2={})={}\%$".format(chr(counter_i + 65), round(ave_tfs_j.min(), 5), threshold_tfs, rsquared_tfs_j),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (Temporal Theil Scaling) ----
                plot_2 = ax2[counter_i].plot(
                    dates_tts_j,
                    exponent_tts_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                #ax2[counter_i].fill_between(
                #    dates_tts_j,
                #    exponent_tts_j - error_exponent_tts_j,
                #    exponent_tts_j + error_exponent_tts_j,
                #    alpha = 0.19,
                #    facecolor = colors[counter_i],
                #    interpolate = True
                #)
                ax2[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax2[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax2[counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax2[counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax2[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax2[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax2[counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                #ax2[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax2[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax2[counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax2[counter_i].set_ylabel("TTS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax2[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax2[counter_i].set_title(
                    r"({}) $MAE_p={}$, $q(R^2={})={}\%$".format(chr(counter_i + 65), round(ave_tts_j.min(), 5), threshold_tts, rsquared_tts_j),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (about Generalized Hurst Exponent) ----
                plot_3 = ax3[counter_i]
                plot_4 = ax4[counter_i]
                plot_5 = ax5[counter_i]
                counter_j = 0
                
                for k in sorted(df_aux["q_order"].unique().tolist()):
                    df_aux_k = df_aux[df_aux["q_order"] == k]
                    
                    # Extract empirical data (All dates, exponents information and others) ----
                    dates_tfs_k = pd.to_datetime(df_aux_k["date"], errors = "coerce")
                    dates_tts_k = pd.to_datetime(df_aux_k["date"], errors = "coerce")
                    exponent_tfs_k = df_aux_k["exponent_tfs"]
                    exponent_tts_k = df_aux_k["exponent_tts"]
                    coefficient_tfs_k = df_aux_k["coefficient_tfs"]
                    coefficient_tts_k = df_aux_k["coefficient_tts"]
                    hurst_k = df_aux_k["hurst"]

                    error_exponent_tfs_k = df_aux_k["error_exponent_tfs"]
                    error_exponent_tts_k = df_aux_k["error_exponent_tts"]
                    error_coefficient_tfs_k = df_aux_k["error_coefficient_tfs"]
                    error_coefficient_tts_k = df_aux_k["error_coefficient_tts"]
                    
                    # Estimation of parameters with theorical value and uncertainty ----
                    try:
                        popt_tfs_k, pcov_tfs_k = np.polyfit(x = exponent_tfs_k.to_numpy(), y = hurst_k.to_numpy(), deg = degree_tfs, cov = True)
                    except Exception as e:
                        popt_tfs_k, pcov_tfs_k = np.zeros(degree_tfs + 1), np.zeros([degree_tfs + 1, degree_tfs + 1]) 
                    
                    try:
                        popt_tts_k, pcov_tts_k = np.polyfit(x = exponent_tts_k.to_numpy(), y = hurst_k.to_numpy(), deg = degree_tts, cov = True)
                    except Exception as e:
                        popt_tts_k, pcov_tts_k = np.zeros(degree_tts + 1), np.zeros([degree_tts + 1, degree_tts + 1])
                    
                    error_tfs_k = np.sqrt(np.diag(pcov_tfs_k)) / np.sqrt(len(exponent_tfs_k))
                    error_tfs_k[np.isinf(error_tfs_k)] = 0
                    lower_tfs_k = popt_tfs_k - error_tfs_k 
                    upper_tfs_k = popt_tfs_k + error_tfs_k

                    error_tts_k = np.sqrt(np.diag(pcov_tts_k)) / np.sqrt(len(exponent_tts_k))
                    error_tts_k[np.isinf(error_tts_k)] = 0
                    lower_tts_k = popt_tts_k - error_tts_k 
                    upper_tts_k = popt_tts_k + error_tts_k

                    estimated_tfs_mean_ = np.poly1d(popt_tfs_k)(exponent_tfs_k)                    
                    estimated_tfs_lower = np.poly1d(lower_tfs_k)(exponent_tfs_k)                 
                    estimated_tfs_upper = np.poly1d(upper_tfs_k)(exponent_tfs_k)  
                    
                    estimated_tts_mean_ = np.poly1d(popt_tts_k)(exponent_tts_k)                    
                    estimated_tts_lower = np.poly1d(lower_tts_k)(exponent_tts_k)                 
                    estimated_tts_upper = np.poly1d(upper_tts_k)(exponent_tts_k)  
                    
                    r2_tfs_k = estimate_coefficient_of_determination(y = hurst_k.values, y_fitted = estimated_tfs_mean_)
                    r2_tts_k = estimate_coefficient_of_determination(y = hurst_k.values, y_fitted = estimated_tts_mean_)
                    ae_tfs_k = estimate_p_norm(x = hurst_k.values, y = estimated_tfs_mean_, p = df_aux_k["p_norm_tfs"].min())        
                    ae_tts_k = estimate_p_norm(x = hurst_k.values, y = estimated_tts_mean_, p = df_aux_k["p_norm_tts"].min())        
                    
                    # Final data of regression ----
                    df_hurst_tfs = df_hurst_tfs.append(
                        pd.DataFrame(
                            {
                                "symbol" : [i] * (degree_tfs + 1),
                                "time_series" : [j] * (degree_tfs + 1),
                                "q_order" : [k] * (degree_tfs + 1),
                                "parameters_tfs" : popt_tfs_k,
                                "error_parameters_tfs" : error_tfs_k,
                                "rsquared_tfs" : [r2_tfs_k] * (degree_tfs + 1),
                                "average_error_tfs" : [ae_tfs_k] * (degree_tfs + 1)
                            }
                        )
                    )
                    
                    df_hurst_tts = df_hurst_tts.append(
                        pd.DataFrame(
                            {
                                "symbol" : [i] * (degree_tts + 1),
                                "time_series" : [j] * (degree_tts + 1),
                                "q_order" : [k] * (degree_tts + 1),
                                "parameters_tts" : popt_tts_k,
                                "error_parameters_tts" : error_tts_k,
                                "rsquared_tts" : [r2_tts_k] * (degree_tts + 1),
                                "average_error_tts" : [ae_tts_k] * (degree_tts + 1)
                            }
                        )
                    )
                    
                    # Plot graph (Hurst exponent) ----
                    ax3[counter_i].plot(
                        dates_tfs_k,
                        hurst_k,
                        alpha = 1,
                        zorder = 2,
                        color = colors[counter_j],
                        marker = "o",
                        linestyle = "",
                        label = "q = {}".format(k),
                        markersize = markersize
                    )
                    
                    # Plot graph (Hurst exponent vs Temporal Fluctuation Scaling) ----
                    ax4[counter_i].plot(
                        exponent_tfs_k,
                        hurst_k,
                        alpha = 1,
                        zorder = 2,
                        color = colors[counter_j],
                        marker = counter_j + 3,
                        linestyle = "",
                        label = "q = {}".format(k),
                        markersize = markersize
                    )
                    if np.unique(estimated_tfs_mean_)[0] != 0:
                        ax4[counter_i].plot(
                            exponent_tfs_k,
                            estimated_tfs_mean_,
                            alpha = 1,
                            zorder = 2,
                            color = colors[counter_j],
                            linewidth = 3,
                            label = "Fit q={}".format(k)
                        )
                        #ax4[counter_i].fill_between(
                        #    exponent_tfs_k,
                        #    estimated_tfs_lower,
                        #    estimated_tfs_upper,
                        #    where = (
                        #        (estimated_tfs_upper >= estimated_tfs_lower) &
                        #        (estimated_tfs_upper >= estimated_tfs_mean_) &
                        #        (estimated_tfs_mean_ >= estimated_tfs_lower)
                        #    ),
                        #    alpha = 0.19,
                        #    facecolor = colors[counter_j],
                        #    interpolate = True
                        #)
                    
                    # Plot graph (Hurst exponent vs Temporal Theil Scaling) ----
                    ax5[counter_i].plot(
                        exponent_tts_k,
                        hurst_k,
                        alpha = 1,
                        zorder = 2,
                        color = colors[counter_j],
                        marker = counter_j + 3,
                        linestyle = "",
                        label = "q = {}".format(k),
                        markersize = markersize
                    )
                    if np.unique(estimated_tts_mean_)[0] != 0:
                        ax5[counter_i].plot(
                            exponent_tts_k,
                            estimated_tts_mean_,
                            alpha = 1,
                            zorder = 2,
                            color = colors[counter_j],
                            linewidth = 3,
                            label = "Fit q={}".format(k)
                        )
                        #ax5[counter_i].fill_between(
                        #    exponent_tfs_k,
                        #    estimated_tts_lower,
                        #    estimated_tts_upper,
                        #    where = (
                        #        (estimated_tts_upper >= estimated_tts_lower) &
                        #        (estimated_tts_upper >= estimated_tts_mean_) &
                        #        (estimated_tts_mean_ >= estimated_tts_lower)
                        #    ),
                        #    alpha = 0.19,
                        #    facecolor = colors[counter_j],
                        #    interpolate = True
                        #)
                
                    counter_j += 1
                
                # Plot graph (Hurst exponent) ----
                ax3[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax3[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax3[counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax3[counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax3[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax3[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax3[counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                #ax3[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax3[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax3[counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax3[counter_i].set_ylabel("Generalized Hurst exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax3[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax3[counter_i].set_title(
                    r"({}) $T_{{MF}}={}$".format(chr(counter_i + 65), threshold_n_data_hurst),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (Hurst exponent vs Temporal Fluctuation Scaling) ----
                ax4[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax4[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax4[counter_i].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
                ax4[counter_i].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
                ax4[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax4[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                #ax4[counter_i].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                #ax4[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax4[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax4[counter_i].set_xlabel("TFS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax4[counter_i].set_ylabel("Generalized Hurst exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax4[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax4[counter_i].set_title(
                    r"({}) $T_{{MF}}={}$, $W={}$, $r_{{TFS}}={}$, $MAE_p={}$, $R^2={}\%$".format(
                        chr(counter_i + 65),
                        threshold_n_data_hurst,
                        degree_tfs,
                        threshold_tfs,
                        max([round(ae_tfs_k.min(), 5), 0]),
                        max([round(r2_tfs_k * 100, 2), 0])
                    ),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (Hurst exponent vs Temporal Theil Scaling) ----
                ax5[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax5[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax5[counter_i].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
                ax5[counter_i].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
                ax5[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax5[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                #ax5[counter_i].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                #ax5[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax5[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax5[counter_i].set_xlabel("TTS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax5[counter_i].set_ylabel("Generalized Hurst exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax5[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax5[counter_i].set_title(
                    r"({}) $T_{{MF}}={}$, $W={}$, $r_{{TTS}}={}$, $MAE_p={}$, $R^2={}\%$".format(
                        chr(counter_i + 65),
                        threshold_n_data_hurst,
                        degree_tts,
                        threshold_tts,
                        max([round(ae_tts_k.min(), 5), 0]),
                        max([round(r2_tts_k * 100, 2), 0])
                    ),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
            else:
                # Plot graph (Temporal Fluctuation Scaling) ----
                plot_1 = ax1[counter, counter_i].plot(
                    dates_tfs_j,
                    exponent_tfs_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                #ax1[counter, counter_i].fill_between(
                #    dates_tfs_j,
                #    exponent_tfs_j - error_exponent_tfs_j,
                #    exponent_tfs_j + error_exponent_tfs_j,
                #    alpha = 0.19,
                #    facecolor = colors[counter_i],
                #    interpolate = True
                #)
                ax1[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax1[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax1[counter, counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax1[counter, counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax1[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax1[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax1[counter, counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                #ax1[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax1[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax1[counter, counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax1[counter, counter_i].set_ylabel("TFS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax1[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax1[counter, counter_i].set_title(
                    r"({}) $MAE_p={}$, $q(R^2={})={}\%$".format(chr(counter_i + 65), round(ave_tfs_j.min(), 5), threshold_tfs, rsquared_tfs_j),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (Temporal Theil Scaling) ----
                plot_2 = ax2[counter, counter_i].plot(
                    dates_tts_j,
                    exponent_tts_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                #ax2[counter, counter_i].fill_between(
                #    dates_tts_j,
                #    exponent_tts_j - error_exponent_tts_j,
                #    exponent_tts_j + error_exponent_tts_j,
                #    alpha = 0.19,
                #    facecolor = colors[counter_i],
                #    interpolate = True
                #)
                ax2[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax2[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax2[counter, counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax2[counter, counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax2[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax2[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax2[counter, counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                #ax2[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax2[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax2[counter, counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax2[counter, counter_i].set_ylabel("TTS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax2[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax2[counter, counter_i].set_title(
                    r"({}) $MAE_p={}$, $q(R^2={})={}\%$".format(chr(counter_i + 65), round(ave_tts_j.min(), 5), threshold_tts, rsquared_tts_j),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (about Generalized Hurst Exponent) ----
                plot_3 = ax3[counter, counter_i]
                plot_4 = ax4[counter, counter_i]
                plot_5 = ax5[counter, counter_i]
                counter_j = 0
                
                for k in sorted(df_aux["q_order"].unique().tolist()):
                    df_aux_k = df_aux[df_aux["q_order"] == k]
                    
                    # Extract empirical data (All dates, exponents information and others) ----
                    dates_tfs_k = pd.to_datetime(df_aux_k["date"], errors = "coerce")
                    dates_tts_k = pd.to_datetime(df_aux_k["date"], errors = "coerce")
                    exponent_tfs_k = df_aux_k["exponent_tfs"]
                    exponent_tts_k = df_aux_k["exponent_tts"]
                    coefficient_tfs_k = df_aux_k["coefficient_tfs"]
                    coefficient_tts_k = df_aux_k["coefficient_tts"]
                    hurst_k = df_aux_k["hurst"]

                    error_exponent_tfs_k = df_aux_k["error_exponent_tfs"]
                    error_exponent_tts_k = df_aux_k["error_exponent_tts"]
                    error_coefficient_tfs_k = df_aux_k["error_coefficient_tfs"]
                    error_coefficient_tts_k = df_aux_k["error_coefficient_tts"]
                    
                    # Estimation of parameters with theorical value and uncertainty ----
                    try:
                        popt_tfs_k, pcov_tfs_k = np.polyfit(x = exponent_tfs_k.to_numpy(), y = hurst_k.to_numpy(), deg = degree_tfs, cov = True)
                    except Exception as e:
                        popt_tfs_k, pcov_tfs_k = np.zeros(degree_tfs + 1), np.zeros([degree_tfs + 1, degree_tfs + 1]) 
                        
                    try:
                        popt_tts_k, pcov_tts_k = np.polyfit(x = exponent_tts_k.to_numpy(), y = hurst_k.to_numpy(), deg = degree_tts, cov = True)
                    except Exception as e:
                        popt_tts_k, pcov_tts_k = np.zeros(degree_tts + 1), np.zeros([degree_tts + 1, degree_tts + 1]) 
                    
                    error_tfs_k = np.sqrt(np.diag(pcov_tfs_k)) / np.sqrt(len(exponent_tfs_k))
                    error_tfs_k[np.isinf(error_tfs_k)] = 0
                    lower_tfs_k = popt_tfs_k - error_tfs_k 
                    upper_tfs_k = popt_tfs_k + error_tfs_k

                    error_tts_k = np.sqrt(np.diag(pcov_tts_k)) / np.sqrt(len(exponent_tts_k))
                    error_tts_k[np.isinf(error_tts_k)] = 0
                    lower_tts_k = popt_tts_k - error_tts_k 
                    upper_tts_k = popt_tts_k + error_tts_k

                    estimated_tfs_mean_ = np.poly1d(popt_tfs_k)(exponent_tfs_k)                    
                    estimated_tfs_lower = np.poly1d(lower_tfs_k)(exponent_tfs_k)                 
                    estimated_tfs_upper = np.poly1d(upper_tfs_k)(exponent_tfs_k)  
                    
                    estimated_tts_mean_ = np.poly1d(popt_tts_k)(exponent_tts_k)                    
                    estimated_tts_lower = np.poly1d(lower_tts_k)(exponent_tts_k)                 
                    estimated_tts_upper = np.poly1d(upper_tts_k)(exponent_tts_k)  
                    
                    r2_tfs_k = estimate_coefficient_of_determination(y = hurst_k.values, y_fitted = estimated_tfs_mean_)
                    r2_tts_k = estimate_coefficient_of_determination(y = hurst_k.values, y_fitted = estimated_tts_mean_)
                    ae_tfs_k = estimate_p_norm(x = hurst_k.values, y = estimated_tfs_mean_, p = df_aux_k["p_norm_tfs"].min())        
                    ae_tts_k = estimate_p_norm(x = hurst_k.values, y = estimated_tts_mean_, p = df_aux_k["p_norm_tts"].min())        
                    
                    # Final data of regression ----
                    df_hurst_tfs = df_hurst_tfs.append(
                        pd.DataFrame(
                            {
                                "symbol" : [i] * (degree_tfs + 1),
                                "time_series" : [j] * (degree_tfs + 1),
                                "q_order" : [k] * (degree_tfs + 1),
                                "parameters_tfs" : popt_tfs_k,
                                "error_parameters_tfs" : error_tfs_k,
                                "rsquared_tfs" : [r2_tfs_k] * (degree_tfs + 1),
                                "average_error_tfs" : [ae_tfs_k] * (degree_tfs + 1)
                            }
                        )
                    )
                    
                    df_hurst_tts = df_hurst_tts.append(
                        pd.DataFrame(
                            {
                                "symbol" : [i] * (degree_tts + 1),
                                "time_series" : [j] * (degree_tts + 1),
                                "q_order" : [k] * (degree_tts + 1),
                                "parameters_tts" : popt_tts_k,
                                "error_parameters_tts" : error_tts_k,
                                "rsquared_tts" : [r2_tts_k] * (degree_tts + 1),
                                "average_error_tts" : [ae_tts_k] * (degree_tts + 1)
                            }
                        )
                    )
                    
                    # Plot graph (Hurst exponent) ----
                    ax3[counter, counter_i].plot(
                        dates_tfs_k,
                        hurst_k,
                        alpha = 1,
                        zorder = 2,
                        color = colors[counter_j],
                        marker = "o",
                        linestyle = "",
                        label = "q = {}".format(k),
                        markersize = markersize
                    )
                    
                    # Plot graph (Hurst exponent vs Temporal Fluctuation Scaling) ----
                    ax4[counter, counter_i].plot(
                        exponent_tfs_k,
                        hurst_k,
                        alpha = 1,
                        zorder = 2,
                        color = colors[counter_j],
                        marker = counter_j + 3,
                        linestyle = "",
                        label = "q = {}".format(k),
                        markersize = markersize
                    )
                    if np.unique(estimated_tfs_mean_)[0] != 0:
                        ax4[counter, counter_i].plot(
                            exponent_tfs_k,
                            estimated_tfs_mean_,
                            alpha = 1,
                            zorder = 2,
                            color = colors[counter_j],
                            linewidth = 3,
                            label = "Fit q={}".format(k)
                        )
                        #ax4[counter, counter_i].fill_between(
                        #    exponent_tfs_k,
                        #    estimated_tfs_lower,
                        #    estimated_tfs_upper,
                        #    where = (
                        #        (estimated_tfs_upper >= estimated_tfs_lower) &
                        #        (estimated_tfs_upper >= estimated_tfs_mean_) &
                        #        (estimated_tfs_mean_ >= estimated_tfs_lower)
                        #    ),
                        #    alpha = 0.19,
                        #    facecolor = colors[counter_j],
                        #    interpolate = True
                        #)
                    
                    # Plot graph (Hurst exponent vs Temporal Theil Scaling) ----
                    ax5[counter, counter_i].plot(
                        exponent_tts_k,
                        hurst_k,
                        alpha = 1,
                        zorder = 2,
                        color = colors[counter_j],
                        marker = counter_j + 3,
                        linestyle = "",
                        label = "q = {}".format(k),
                        markersize = markersize
                    )
                    if np.unique(estimated_tts_mean_)[0] != 0:
                        ax5[counter, counter_i].plot(
                            exponent_tts_k,
                            estimated_tts_mean_,
                            alpha = 1,
                            zorder = 2,
                            color = colors[counter_j],
                            linewidth = 3,
                            label = "Fit q={}".format(k)
                        )
                        #ax5[counter, counter_i].fill_between(
                        #    exponent_tfs_k,
                        #    estimated_tts_lower,
                        #    estimated_tts_upper,
                        #    where = (
                        #        (estimated_tts_upper >= estimated_tts_lower) &
                        #        (estimated_tts_upper >= estimated_tts_mean_) &
                        #        (estimated_tts_mean_ >= estimated_tts_lower)
                        #    ),
                        #    alpha = 0.19,
                        #    facecolor = colors[counter_j],
                        #    interpolate = True
                        #)
                
                    counter_j += 1
                
                # Plot graph (Hurst exponent) ----
                ax3[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax3[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax3[counter, counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax3[counter, counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax3[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax3[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax3[counter, counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                #ax3[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax3[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax3[counter, counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax3[counter, counter_i].set_ylabel("Generalized Hurst exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax3[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax3[counter, counter_i].set_title(
                    r"({}) $T_{{MF}}={}$".format(chr(counter_i + 65), threshold_n_data_hurst),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (Hurst exponent vs Temporal Fluctuation Scaling) ----
                ax4[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax4[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax4[counter, counter_i].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
                ax4[counter, counter_i].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
                ax4[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax4[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                #ax4[counter, counter_i].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                #ax4[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax4[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax4[counter, counter_i].set_xlabel("TFS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax4[counter, counter_i].set_ylabel("Generalized Hurst exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax4[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax4[counter, counter_i].set_title(
                    r"({}) $T_{{MF}}={}$, $W={}$, $r_{{TFS}}={}$, $MAE_p={}$, $R^2={}\%$".format(
                        chr(counter_i + 65),
                        threshold_n_data_hurst,
                        degree_tfs,
                        threshold_tfs,
                        max([round(ae_tfs_k.min(), 5), 0]),
                        max([round(r2_tfs_k * 100, 2), 0])
                    ),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
                
                # Plot graph (Hurst exponent vs Temporal Theil Scaling) ----
                ax5[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax5[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax5[counter, counter_i].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
                ax5[counter, counter_i].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
                ax5[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax5[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                #ax5[counter, counter_i].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                #ax5[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax5[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax5[counter, counter_i].set_xlabel("TTS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax5[counter, counter_i].set_ylabel("Generalized Hurst exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax5[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax5[counter, counter_i].set_title(
                    r"({}) $T_{{MF}}={}$, $W={}$, $r_{{TTS}}={}$, $MAE_p={}$, $R^2={}\%$".format(
                        chr(counter_i + 65),
                        threshold_n_data_hurst,
                        degree_tts,
                        threshold_tts,
                        max([round(ae_tts_k.min(), 5), 0]),
                        max([round(r2_tts_k * 100, 2), 0])
                    ),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
            
            # Function development ----
            counter_i += 1
            print("Generated plot for {} and time series {}".format(i, j))
        
        counter += 1
    
    # final output and plotting
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    if save_figures:
        # Plot graph (Temporal Fluctuation Scaling) ----
        plt.show()
        fig1.savefig(
            "{}/{}_tfs_exponent_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig1.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        
        # Plot graph (Temporal Theil Scaling) ----
        plt.show()
        fig2.savefig(
            "{}/{}_tts_exponent_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig2.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        
        # Plot graph (Hurst exponent) ----
        plt.show()
        fig3.savefig(
            "{}/{}_hurst_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig3.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        
        # Plot graph (Hurst exponent vs Temporal Fluctuation Scaling) ----
        plt.show()
        fig4.savefig(
            "{}/{}_hurst_tfs_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig4.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        
        # Plot graph (Hurst exponent vs Temporal Theil Scaling) ----
        plt.show()
        fig5.savefig(
            "{}/{}_hurst_tts_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig5.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    
    return df_hurst_tfs, df_hurst_tts
