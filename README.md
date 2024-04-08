# Generalized Shannon Index
Multifractality is a concept that extends locally the usual ideas of fractality in a system. Nevertheless, the multifractal approaches used lack a multifractal dimension tied to an entropy index like the Shannon index. This paper introduces a generalized Shannon index (GSI) and demonstrates its application in understanding system fluctuations. To this end, traditional multifractality approaches are explained. Then, using the temporal Theil scaling and the diffusive trajectory algorithm, the GSI and its partition function are defined. Next, the multifractal exponent of the GSI is derived from the partition function, establishing a connection between the temporal Theil scaling exponent and the generalized Hurst exponent. Finally, this relationship is verified in a fractional Brownian motion and applied to financial time series. This leads us to propose an approximation denominated _local approximation of fractional Brownian motion (LA-fBm)_, where multifractal systems are viewed as a local superposition of distinct fractional Brownian motions with varying monofractal exponents. Also, we furnish an algorithm for identifying the optimal $q$-th moment of the probability distribution associated with an empirical time series to enhance the accuracy of generalized Hurst exponent estimation.

## Fractional Brownian motion (fBm)
To estimate a relationship between the temporal Theil scaling exponent $\alpha_{TTS}(t)$ and the generalized Hurst exponent $H(q)$, multiple simulations of fractional Brownian motion (fBm) are generated to which the Multifractal Detrended Fluctuation Analysis (MF-DFA) method is applied to verify the value of the Hurst exponent. Furthermore, the diffusive trajectory algorithm is applied to estimate the temporal Theil scaling exponent $\alpha_{TTS}(t)$ of an fBm. Finally, the linear relationship that exists between the temporal Theil scaling exponent $\alpha_{TTS}(t)$ and the generalized Hurst exponent $H(q)$ is verified. To do this, you have the following scripts:

1. estimate_hurst_tts_relation_fbm
2. plot_hurst_tts_relation

## Financial time series (fts)
To show the relationship between the temporal Theil scaling exponent $\alpha_{TTS}(t)$ and the generalized Hurst exponent $H(q)$ in an arbitrary empirical time series, the Dow Jones and Euro to Colombian peso financial time series are used as an example. From these results, a method is implemented to calculate the most optimal value of the $q$-th moment within a set of test values $q_{1}$, $q_{2}$, ..., $q_{W}$. To do this, you have the following scripts:

1. process_data_fts
2. estimate_hurst_exponent_mfdfa_fts
3. estimate_tfs_fts
4. estimate_diffusive_algorithm_fts
5. estimate_tts_fts
6. prepare_exponents_data
7. compare_exponents
