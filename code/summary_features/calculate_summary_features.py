from scipy import stats as spstats
from scipy.stats import moment
import numpy as np
from scipy.signal import argrelextrema
import torch
import tsfel
from numpy import trapz


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


        # sets the first value as baseline
        #batch = torch.sub(batch, torch.index_select(batch, 0, torch.tensor([0])))

        ##search for P50 between 0 and 80ms:

        #print('batch shape ', batch.shape)

        arg80ms = int(80 * 30)

        arg_p50 = torch.argmax(batch[0:arg80ms])
        arg_P200 = torch.argmax(batch)

        #print('arg_p50', arg_p50)
        #print('arg_p200', arg_P200)

        ## search for N100
        arg_N100 = torch.argmin(batch)

        arg200ms = int(200 * 30)
        arg50ms = int(30* 50)
        arg70ms = int(30 * 70)
        arg100ms = int(30 * 100)
        arg120ms = int(30 * 120)


        if number_stats == 6:
            '''
            - arg_p50: searches for the time of the first postive peak (searches argmax during the first 70ms)
            - arg_N100: searches for the time of the negative peak of the ERP signal (seaches argmin)
            - arg_P200: searches for the time of the second positive peak (searches argmax starting from 70ms to end of signal)
            - p50: value of first postive peak
            - N100: value of negative peak
            - P200: value of second postive peak
            '''

            N100 = torch.min(batch)
            p50 = torch.max(batch[0:arg80ms])
            P200 = torch.max(batch[arg80ms:])

            sum_stats_vec = torch.stack([arg_p50, arg_N100, arg_P200, p50, N100, P200])
            batch_list.append(sum_stats_vec)


        elif number_stats == 17:
            '''
            - arg_p50: searches for the time of the first postive peak (searches argmax during the first 70ms)
            - arg_N100: searches for the time of the negative peak of the ERP signal (seaches argmin)
            - arg_P200: searches for the time of the second positive peak (searches argmax starting from 70ms to end of signal)
            - p50: value of first postive peak
            - N100: value of negative peak
            - P200: value of second postive peak
            - p50_moment1: first moment (mean) around the first postive peak (10ms before, 10ms after),
            - N100_moment1: first moment (mean) around the negative peak (10ms before, 10ms after),,
            - P200_moment1: first moment around the second postive peak (10ms before, 10ms after)
            '''

            N100 = torch.min(batch)
            p50 = torch.max(batch[0:arg80ms])
            P200 = torch.max(batch[arg80ms:])

            p50_moment1 = torch.mean(batch[arg_p50-10*time_window:arg_p50+10*time_window])  # mean
            N100_moment1 = torch.mean(batch[arg_N100-10*time_window:arg200ms+10*time_window])  # mean
            P200_moment1 = torch.mean(batch[arg_P200-10*time_window:arg_P200+10*time_window])  # mean

            p50_moment2 = torch.var(batch[arg_p50-10*time_window:arg_p50+10*time_window])  # variance
            N100_moment2 = torch.var(batch[arg_N100-10*time_window:arg200ms+10*time_window])  # variance
            P200_moment2 = torch.var(batch[arg_P200-10*time_window:arg_P200+10*time_window])  # variance


            ## search zero crossing after p50:
            #print('zero crossings', np.where(np.diff(np.sign(batch)))[0])

            zero_crossings = np.where(np.diff(np.sign(batch)))[0]

            number_crossings = len(np.where(np.diff(np.sign(batch)))[0])
            
            
            zero_cross_p50 = int(zero_crossings[number_crossings-2])
         

            #print('zero crossing p50', zero_cross_p50)

            # compute area under the curve:
            area_p50 = trapz(batch[:zero_cross_p50], dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area_p50 = torch.tensor(area_p50)


            ## search zero crossing after N100:
            #print('arg_N100', arg_N100)
            #print('arg_P200', arg_P200)

     
            zero_cross_N100 = int(zero_crossings[number_crossings-1])

            #print('zero crossing N100', zero_cross_N100)

            # compute area under the curve:
            area_N100 = trapz(batch[zero_cross_p50:zero_cross_N100], dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area_N100 = torch.tensor(area_N100)



            # compute area under the curve:
            area_P200 = trapz(batch[zero_cross_N100:], dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area_P200 = torch.tensor(area_P200)

            # mean values over different, relevant time periods:

            mean4000 = torch.mean(batch[4000:4500])

            mean1000 = torch.mean(batch[:1000])



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
                    area_p50,
                    area_N100,
                    area_P200,
                    mean4000,
                    mean1000
                ]
            )
            batch_list.append(sum_stats_vec)
            

        elif number_stats == 21:
            '''
            - max: maximum over whole time series
            - min: minimum over whole time series
            - peak_to_peak: distance between two succeeding peaks (over whole time series)
            - area: area under the curve (ove whole time series)
            - autocorr: autocorrelation (over whole time series)
            - zero_cross: number of times that signal is crossing zero (over whole time series)
            - max1, min1, peak_to_peak1, area1, autocorr1 (same as before but considering time window 1 - from 0ms to 70ms)
            - max2, min2, peak_to_peak2, area2, autocorr2 (same as before but considering time window 2 - from 50ms to 120ms)
            - max3, min3, peak_to_peak3, area3, autocorr3 (same as before but considering time window 3 - from 100ms)
            '''

            ## idea: overlapping sliding window. we calculate summary statistics for each 'window' seperately.


            window = batch
        
            max = torch.max(window)
            min = torch.min(window)
            peak_to_peak = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:

            
            area = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area = torch.tensor(area)
            ## autocorrelation:
            autocorr = torch.tensor(tsfel.feature_extraction.features.autocorr(window))    

            ## number of times that signal crosses the zero axis:
            zero_cross = torch.tensor(len(np.where(np.diff(np.sign(window)))[0]))       
            


            window = batch[:arg70ms]

            max1 = torch.max(window)
            min1 = torch.min(window)
            peak_to_peak1 = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:

            area1 = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area1 = torch.tensor(area1)
            ## autocorrelation:
            autocorr1 = torch.tensor(tsfel.feature_extraction.features.autocorr(window))
            #frequencies = tsfel.feature_extraction.features.fft_mean_coeff(batch, fs=30, nfreq=256)


            window = batch[arg50ms:arg120ms]

            max2 = torch.max(window)
            min2 = torch.min(window)
            peak_to_peak2 = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:

            area2 = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area2 = torch.tensor(area2)
            ## autocorrelation:
            autocorr2 = torch.tensor(tsfel.feature_extraction.features.autocorr(window))
            #frequencies = tsfel.feature_extraction.features.fft_mean_coeff(batch, fs=30, nfreq=256)


            window = batch[arg100ms:]

            max3 = torch.max(window)
            min3 = torch.min(window)
            peak_to_peak3 = torch.abs(torch.max(window) - torch.min(window))

            # compute area under the curve:
            area3 = trapz(window, dx=1)   # Integrate along the given axis using the composite trapezoidal rule. dx is the spacing between sample points
            area3 = torch.tensor(area3)
            ## autocorrelation:
            autocorr3 = torch.tensor(tsfel.feature_extraction.features.autocorr(window))
            #frequencies = tsfel.feature_extraction.features.fft_mean_coeff(batch, fs=30, nfreq=256)


            sum_stats_vec = torch.stack([max, min, peak_to_peak, area, autocorr, zero_cross, 
                                        max1, min1, peak_to_peak1, area1, autocorr1, 
                                        max2, min2, peak_to_peak2, area2, autocorr2, 
                                        max3, min3, peak_to_peak3, area3, autocorr3])
            batch_list.append(sum_stats_vec)


        elif number_stats == 18:
            N100 = torch.min(batch[:arg200ms])
            p50 = torch.max(batch[0:arg70ms])
            P200 = torch.max(batch[arg70ms:])

            p50_moment1 = torch.tensor(moment(batch[arg_p50-10*time_window:arg_p50+10*time_window], moment=1))  # mean
            N100_moment1 = torch.tensor(moment(batch[arg_N100-10*time_window:arg200ms+10*time_window], moment=1))  # mean
            P200_moment1 = torch.tensor(moment(batch[arg_P200-10*time_window:arg_P200+10*time_window], moment=1))  # mean

            p50_moment2 = torch.tensor(moment(batch[arg_p50-10*time_window:arg_p50+10*time_window], moment=2))  # variance
            N100_moment2 = torch.tensor(moment(batch[arg_N100-10*time_window:arg200ms+10*time_window], moment=2))  # variance
            P200_moment2 = torch.tensor(moment(batch[arg_P200-10*time_window:arg_P200+10*time_window], moment=2))  # variance

            p50_moment3 = torch.tensor(moment(batch[arg_p50-10*time_window:arg_p50+10*time_window], moment=3))  # skewness
            N100_moment3 = torch.tensor(moment(batch[arg_N100-10*time_window:arg200ms+10*time_window], moment=3))  # skewness
            P200_moment3 = torch.tensor(moment(batch[arg_P200-10*time_window:arg_P200+10*time_window], moment=3))  # skewness

            p50_moment4 = torch.tensor(moment(batch[arg_p50-10*time_window:arg_p50+10*time_window], moment=4))  # kurtosis
            N100_moment4 = torch.tensor(moment(batch[arg_N100-10*time_window:arg200ms+10*time_window], moment=4))  # kurtosis
            P200_moment4 = torch.tensor(moment(batch[arg_P200-10*time_window:arg_P200+10*time_window], moment=4))  # kurtosis

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
                    p50_moment3,
                    N100_moment3,
                    P200_moment3,
                    p50_moment4,
                    N100_moment4,
                    P200_moment4,
                ]
            )
            batch_list.append(sum_stats_vec)


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


    time_window = 30

    batch_list = []

    if (x.dim()==1):
        print('single observation')
        x = x.unsqueeze(0)
        print('x shape', x.shape)
    for batch in x:

        print('batch shape', batch.shape)


        total_steps_ms = batch.size(dim=0) / time_window


        arg200ms = int(200 * 30)

        arg120ms = int(120 * 30)

        arg_p50 = torch.argmax(batch[0:arg120ms])


        print('batch shape', batch.shape)
        p50 = torch.max(batch[0:arg120ms])

        p50_moment1 = torch.mean(batch[arg_p50-10*time_window:arg_p50+10*time_window])  # mean

        p50_moment2 = torch.var(batch[arg_p50-10*time_window:arg_p50+10*time_window])  # variance

        ## define area under the curve with respect to the x-axis, and only up to the early stopping of step1
        #x_t = batch[:90*30]
        #sign_x = np.sign(x_t)

        #index_pos = np.where(sign_x == 1)
        #index_neg = np.where(sign_x == -1)

        #area_pos1 = torch.trapz(x_t[index_pos])
        #area_neg1 = torch.trapz(x_t[index_neg])



        # mean values over different, relevant time periods:
        mean1000 = torch.mean(batch[:1000]) 

        ### trying to catch values of second bump:
        mean1500 = torch.mean(batch[1500:1700])
        mean1700 = torch.mean(batch[1700:1900])   
        mean1900 = torch.mean(batch[1900:2100]) 
        mean2100 = torch.mean(batch[2100:2300]) 
        mean2300 = torch.mean(batch[2300:2500]) 
               
        

        if (total_steps_ms < 100):

            sum_stats_vec = torch.stack(
                                [
                    arg_p50,
                    p50,
                    p50_moment1,
                    p50_moment2,
                    #area_pos1,
                    #area_neg1,
                    mean1000,
                    mean1500,
                    mean1700,
                    mean1900,
                    mean2100,
                    mean2300
                ]
            )

            batch_list.append(sum_stats_vec)

            continue

        ## search for N100
        arg_N100 = torch.argmin(batch)

        N100 = torch.min(batch)
        N100_moment1 = torch.mean(batch[arg_N100-10*time_window:arg200ms+10*time_window])  # mean
        N100_moment2 = torch.var(batch[arg_N100-10*time_window:arg200ms+10*time_window])  # variance


        #x_t = batch[90*30:160*30]
        #sign_x = np.sign(x_t)

        #index_pos = np.where(sign_x == 1)
        #index_neg = np.where(sign_x == -1)

        #area_pos2 = torch.trapz(x_t[index_pos])
        #area_neg2 = torch.trapz(x_t[index_neg])

        if (total_steps_ms<170):
            sum_stats_vec = torch.stack(
                [
                    arg_p50,
                    arg_N100,
                    p50,
                    N100,
                    p50_moment1,
                    N100_moment1,
                    N100_moment2,
                    mean1000,
                    mean1500,
                    mean1700,
                    mean1900,
                    mean2100,
                    mean2300,
                ]
            )

            batch_list.append(sum_stats_vec)

            continue

        mean4000 = torch.mean(batch[4000:4500])

        mean6000 = torch.mean(batch[6000:700])
        mean7000 = torch.mean(batch[7000:])

        arg_P200 = torch.argmax(batch)

        P200 = torch.max(batch[arg120ms:])

        P200_moment1 = torch.mean(batch[arg_P200-10*time_window:arg_P200+10*time_window])  # mean
        P200_moment2 = torch.var(batch[arg_P200-10*time_window:arg_P200+10*time_window])  # variance

        #x_t = batch[arg120ms:]
        #sign_x = np.sign(x_t)

        #index_pos = np.where(sign_x == 1)
        #index_neg = np.where(sign_x == -1)

        #area_pos3 = torch.trapz(x_t[index_pos])
        #area_neg3 = torch.trapz(x_t[index_neg])
        
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
                    mean1000,
                    mean1500,
                    mean1700,
                    mean1900,
                    mean2100,
                    mean2300,
                    mean4000,
                    mean6000,
                    mean7000,
                ])

        batch_list.append(sum_stats_vec)




    sum_stats = torch.stack(batch_list)

    return sum_stats


def calculate_summary_statistics_alternative(x, step=40):
    """
    reduces time resolution, but does not calculate real summary statistics
    with x[:,::20] every 20th step is taken into account. there is no kind of interpolation
    """
    print('x', x)
    if (x.dim()==2):
        sum_stat = x[:,::step]
        print('sum stats shape', sum_stat.shape)
    else:
        sum_stat = x[::step]
        print('sum stats shape', sum_stat.shape)
    return sum_stat

