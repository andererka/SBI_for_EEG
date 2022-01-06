from scipy import stats as spstats
from scipy.stats import moment
import numpy as np
from scipy.signal import argrelextrema
import torch
import tsfel


def calculate_summary_stats_number(x, number_stats):
    """
    Input: observations or simulations
    Returns summary statistics 
    specific for event related potentials
    extracts the following:
     P50: time, value, mean, variance, skewness and kurtosis of the time interval around it
     N100: time, value, mean, variance, skewness and kurtosis of the time interval around it
     P200: time, value, mean, variance, skewness and kurtosis of the time interval around it
    """

    time_window = 30

    batch_list = []

    for batch in x:

        total_steps_ms = batch.size(dim=0) / time_window

        # sets the first value as baseline
        batch = torch.sub(batch, torch.index_select(batch, 0, torch.tensor([0])))

        ##search for P50 between 0 and 70ms:

        arg70ms = int(np.round(batch.size(dim=0) / total_steps_ms * 70))

        arg_p50 = torch.argmax(batch[0:arg70ms])
        arg_P200 = torch.argmax(batch[arg70ms:])

        ## search for N100
        arg200ms = int(np.round(batch.size(dim=0) / total_steps_ms * 200))
        arg_N100 = torch.argmin(batch[:arg200ms])

        if number_stats == 3:
            sum_stats_vec = torch.stack([arg_p50, arg_N100, arg_P200])
            batch_list.append(sum_stats_vec)

        elif number_stats == 6:

            N100 = torch.min(batch[:arg200ms])
            p50 = torch.max(batch[0:arg70ms])
            P200 = torch.max(batch[arg70ms:])

            sum_stats_vec = torch.stack([arg_p50, arg_N100, arg_P200, p50, N100, P200])
            batch_list.append(sum_stats_vec)

        elif number_stats == 18:

            ## idea: overlapping sliding window. we calculate summary statistics for each 'window' seperately.
            arg50ms = int(np.round(batch.size(dim=0) / total_steps_ms * 50))
            arg100ms = int(np.round(batch.size(dim=0) / total_steps_ms * 100))
            arg150ms = int(np.round(batch.size(dim=0) / total_steps_ms * 150))

            window = batch[:arg100ms]
            print('window', window)
            max1 = torch.max(window)
            min1 = torch.min(window)
            peak_to_peak1 = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:

            from numpy import trapz
            area1 = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area1 = torch.from_numpy(area1)
            ## autocorrelation:
            autocorr1 = tsfel.feature_extraction.features.autocorr(window)
            #frequencies = tsfel.feature_extraction.features.fft_mean_coeff(batch, fs=30, nfreq=256)

            ## number of times that signal crosses the zero axis:
            zero_cross1 = len(np.where(np.diff(np.sign(window)))[0])


            window = batch[arg50ms:arg150ms]

            max2 = torch.max(window)
            min2 = torch.min(window)
            peak_to_peak2 = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:

            from numpy import trapz
            area2 = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area2 = torch.from_numpy(area2)
            ## autocorrelation:
            autocorr2 = tsfel.feature_extraction.features.autocorr(window)
            #frequencies = tsfel.feature_extraction.features.fft_mean_coeff(batch, fs=30, nfreq=256)

            ## number of times that signal crosses the zero axis:
            zero_cross2 = len(np.where(np.diff(np.sign(window)))[0])

            window = batch[arg100ms:]

            max3 = torch.max(window)
            min3 = torch.min(window)
            peak_to_peak3 = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:

            from numpy import trapz
            area3 = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points

            ## autocorrelation:
            autocorr3 = tsfel.feature_extraction.features.autocorr(window)
            #frequencies = tsfel.feature_extraction.features.fft_mean_coeff(batch, fs=30, nfreq=256)

            ## number of times that signal crosses the zero axis:
            zero_cross3 = len(np.where(np.diff(np.sign(window)))[0])


            sum_stats_vec = torch.stack([max1, min1, peak_to_peak1, area1, autocorr1, zero_cross1, 
                                        max2, min2, peak_to_peak2, area2, autocorr2, zero_cross2,
                                        max3, min3, peak_to_peak3, area3, autocorr3, zero_cross3])
            batch_list.append(sum_stats_vec)

        elif number_stats == 9:

            N100 = torch.min(batch[:arg200ms])
            p50 = torch.max(batch[0:arg70ms])
            P200 = torch.max(batch[arg70ms:])

            p50_moment1 = torch.tensor(moment(batch[0:arg70ms], moment=1))  # mean
            N100_moment1 = torch.tensor(moment(batch[0:arg200ms], moment=1))  # mean
            P200_moment1 = torch.tensor(moment(batch[arg70ms:], moment=1))  # mean

            sum_stats_vec = torch.stack(
                [
                    arg_p50,
                    arg_N100,
                    arg_P200,
                    p50,
                    N100,
                    P200,
                    p50_moment1,
                    N100_moment1,
                    P200_moment1,
                ]
            )
            batch_list.append(sum_stats_vec)

        elif number_stats == 12:
            N100 = torch.min(batch[:arg200ms])
            p50 = torch.max(batch[0:arg70ms])
            P200 = torch.max(batch[arg70ms:])

            p50_moment1 = torch.tensor(moment(batch[0:arg70ms], moment=1))  # mean
            N100_moment1 = torch.tensor(moment(batch[0:arg200ms], moment=1))  # mean
            P200_moment1 = torch.tensor(moment(batch[arg70ms:], moment=1))  # mean

            N100_moment2 = torch.tensor(moment(batch[0:arg200ms], moment=2))  # variance
            P200_moment2 = torch.tensor(moment(batch[arg70ms:], moment=2))  # variance
            p50_moment2 = torch.tensor(moment(batch[0:arg70ms], moment=2))  # variance

            sum_stats_vec = torch.stack(
                [
                    arg_p50,
                    arg_N100,
                    arg_P200,
                    p50,
                    N100,
                    P200,
                    p50_moment1,
                    N100_moment1,
                    P200_moment1,
                    p50_moment2,
                    N100_moment2,
                    P200_moment2,
                ]
            )
            batch_list.append(sum_stats_vec)

        # N100_moment3 = torch.tensor(moment(x[0:arg200ms], moment=3))    #skewness
        # P200_moment3 = torch.tensor(moment(x[arg70ms:], moment=3))   #skewness
        # p50_moment3 = torch.tensor(moment(x[0:arg70ms], moment=3) )  #skewness

        # N100_moment4 = torch.tensor(moment(x[0:arg200ms], moment=4))    #kurtosis
        # P200_moment4 = torch.tensor(moment(x[arg70ms:], moment=4))    #kurtosis
        # p50_moment4 = torch.tensor(moment(x[0:arg70ms], moment=4) )  #kurtosis

    sum_stats = torch.stack(batch_list)

    return sum_stats



### these are the summary functions for the sequential approach:


def calculate_summary_stats_temporal(x):
    """
    Input: observations or simulations
    Returns summary statistics 
    specific for event related potentials
    extracts the following:
     P50: time, value, mean, variance, skewness and kurtosis of the time interval around it

    """

    print('len of x in sum stat function:', len(x))
    time_window = 30

    batch_list = []
    print("x", x)

    for batch in x:

        print("batch", batch)
        print('batch size', batch.size(dim=0))
        total_steps_ms = batch.size(dim=0) / time_window

        print("total steps in ms", total_steps_ms)

        # sets the first value as baseline
        batch = torch.sub(batch, torch.index_select(batch, 0, torch.tensor([0])))

        ##search for P50 between 0 and 70ms:

        arg70ms = int(np.round(batch.size(dim=0) / total_steps_ms * 70))
        p50 = torch.max(batch[0:arg70ms])

        arg_p50 = torch.argmax(batch[0:arg70ms])
        p50_moment1 = torch.tensor(
            moment(batch[0:arg70ms], moment=1), dtype=torch.float32
        )  # mean
        # p50_moment2 = torch.tensor(moment(x[0:arg70ms], moment=2) )   #variance
        # p50_moment3 = torch.tensor(moment(x[0:arg70ms], moment=3) )  #skewness
        # p50_moment4 = torch.tensor(moment(x[0:arg70ms], moment=4) )  #kurtosis


        if (total_steps_ms < 70):
            sum_stats_vec = torch.stack([p50, arg_p50, p50_moment1,])

            batch_list.append(sum_stats_vec)

            print('here')

            continue

        print('shouldnt be here')
        ## search for N100
        arg200ms = int(np.round(batch.size(dim=0) / total_steps_ms * 200))

        N100 = torch.min(batch[:arg200ms])

        arg_N100 = torch.argmin(batch[:arg200ms])

        N100_moment1 = torch.tensor(
            moment(batch[0:arg200ms], moment=1), dtype=torch.float32
        )  # mean


        if (total_steps_ms<200):
            sum_stats_vec = torch.stack(
                [
                    p50, 
                    arg_p50, 
                    p50_moment1,
                    N100,
                    arg_N100,
                    N100_moment1,  
                ]
            )

            batch_list.append(sum_stats_vec)

            continue

        P200 = torch.max(batch[arg70ms:])

        arg_P200 = torch.argmax(batch[arg70ms:])
        P200_moment1 = torch.tensor(
            moment(batch[arg70ms:], moment=1), dtype=torch.float32
        )  # mean

        sum_stats_vec = torch.stack(
            [
                p50, 
                arg_p50, 
                p50_moment1,
                N100,
                arg_N100,
                N100_moment1,  
                P200, 
                arg_P200, 
                P200_moment1] 
        )

        batch_list.append(sum_stats_vec)




    sum_stats = torch.stack(batch_list)

    return sum_stats


def calculate_summary_statistics_alternative(x, number=0):
    """
    reduces time resolution, but does not calculate real summary statistics
    with x[:,::20] every 20th step is taken into account. there is no kind of interpolation
    """
    if (x.dim()==2):
        print(x.shape)
        sum_stat = x[:,::20]
        print(sum_stat.shape)
    else:
        sum_stat = x[::20]
        print(sum_stat.shape)
    return sum_stat

