from scipy import stats as spstats
from scipy.stats import moment
import numpy as np
from scipy.signal import argrelextrema
import torch




def calculate_summary_stats(x):
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
        print(batch)
        print(batch.size())
        print(type(batch))

        total_steps_ms = batch.size(dim=0)/time_window

        #sets the first value as baseline
        batch = torch.sub(batch, torch.index_select(batch,0,torch.tensor([0])))


        ##search for P50 between 0 and 70ms:

        arg70ms = int(np.round(batch.size(dim=0) /total_steps_ms *70 ))
        p50 = torch.max(batch[0:arg70ms])

        arg_p50 = torch.argmax(batch[0:arg70ms])
        #p50_moment1 = torch.tensor(moment(x[0:arg70ms], moment=1) )  #mean
        #p50_moment2 = torch.tensor(moment(x[0:arg70ms], moment=2) )   #variance
        #p50_moment3 = torch.tensor(moment(x[0:arg70ms], moment=3) )  #skewness
        #p50_moment4 = torch.tensor(moment(x[0:arg70ms], moment=4) )  #kurtosis

        ## search for N100 
        arg200ms = int(np.round(batch.size(dim=0) /total_steps_ms *200))


        N100 = torch.min(batch[:arg200ms])

        arg_N100 = torch.argmin(batch[:arg200ms])
        #N100_moment1 = torch.tensor(moment(x[0:arg200ms], moment=1))   #mean
        #N100_moment2 = torch.tensor(moment(x[0:arg200ms], moment=2))    #variance
        #N100_moment3 = torch.tensor(moment(x[0:arg200ms], moment=3))    #skewness
        #N100_moment4 = torch.tensor(moment(x[0:arg200ms], moment=4))    #kurtosis


        P200 = torch.max(batch[arg70ms:])

        arg_P200 = torch.argmax(batch[arg70ms:])
        #P200_moment1 = torch.tensor(moment(x[arg70ms:], moment=1))   #mean
        #P200_moment2 = torch.tensor(moment(x[arg70ms:], moment=2))    #variance
        #P200_moment3 = torch.tensor(moment(x[arg70ms:], moment=3))   #skewness
        #P200_moment4 = torch.tensor(moment(x[arg70ms:], moment=4))    #kurtosis



        
        sum_stats_vec = torch.stack([p50, N100, P200, arg_p50, 
        arg_N100, arg_P200,
        #p50_moment1, p50_moment2, p50_moment3, #p50_moment4,
        #N100_moment1, N100_moment2, N100_moment3, #N100_moment4,
        #P200_moment1,P200_moment2, P200_moment3, #P200_moment4
                ])

        #print(sum_stats_list)
        #sum_stats_vec = torch.stack(sum_stats_list)

        batch_list.append(sum_stats_vec)

    sum_stats = torch.stack(batch_list)

    print(sum_stats.size())

    
    return sum_stats


def calculate_summary_stats9(x):
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
      
        total_steps_ms = batch.size(dim=0)/time_window

        #sets the first value as baseline
        batch = torch.sub(batch, torch.index_select(batch,0,torch.tensor([0])))


        ##search for P50 between 0 and 70ms:

        arg70ms = int(np.round(batch.size(dim=0) /total_steps_ms *70 ))
        p50 = torch.max(batch[0:arg70ms])

        arg_p50 = torch.argmax(batch[0:arg70ms])
        p50_moment1 = torch.tensor(moment(batch[0:arg70ms], moment=1) , dtype=torch.float32)  #mean
        #p50_moment2 = torch.tensor(moment(x[0:arg70ms], moment=2) )   #variance
        #p50_moment3 = torch.tensor(moment(x[0:arg70ms], moment=3) )  #skewness
        #p50_moment4 = torch.tensor(moment(x[0:arg70ms], moment=4) )  #kurtosis

        ## search for N100 
        arg200ms = int(np.round(batch.size(dim=0) /total_steps_ms *200))


        N100 = torch.min(batch[:arg200ms])

        arg_N100 = torch.argmin(batch[:arg200ms])
        N100_moment1 = torch.tensor(moment(batch[0:arg200ms], moment=1), dtype=torch.float32)   #mean
        #N100_moment2 = torch.tensor(moment(x[0:arg200ms], moment=2))    #variance
        #N100_moment3 = torch.tensor(moment(x[0:arg200ms], moment=3))    #skewness
        #N100_moment4 = torch.tensor(moment(x[0:arg200ms], moment=4))    #kurtosis


        P200 = torch.max(batch[arg70ms:])

        arg_P200 = torch.argmax(batch[arg70ms:])
        P200_moment1 = torch.tensor(moment(batch[arg70ms:], moment=1), dtype=torch.float32)   #mean
        #P200_moment2 = torch.tensor(moment(x[arg70ms:], moment=2))    #variance
        #P200_moment3 = torch.tensor(moment(x[arg70ms:], moment=3))   #skewness
        #P200_moment4 = torch.tensor(moment(x[arg70ms:], moment=4))    #kurtosis



        
        sum_stats_vec = torch.stack([p50, N100, P200, arg_p50, 
        arg_N100, arg_P200,
        p50_moment1, #p50_moment2, p50_moment3, #p50_moment4,
        N100_moment1, #N100_moment2, N100_moment3, #N100_moment4,
        P200_moment1, #P200_moment2, P200_moment3, #P200_moment4
                ])

        #print(sum_stats_list)
        #sum_stats_vec = torch.stack(sum_stats_list)

        batch_list.append(sum_stats_vec)

    sum_stats = torch.stack(batch_list)

    print('sum stats', type(sum_stats))
    print(sum_stats)

    
    return sum_stats



### these are the summary functions for the sequential approach:

def calculate_summary_stats_P50(x):
    """
    Input: observations or simulations
    Returns summary statistics 
    specific for event related potentials
    extracts the following:
     P50: time, value, mean, variance, skewness and kurtosis of the time interval around it

    """
  
    time_window = 30

    batch_list = []

    for batch in x:
   

        total_steps_ms = batch.size(dim=0)/time_window

        #sets the first value as baseline
        batch = torch.sub(batch, torch.index_select(batch,0,torch.tensor([0])))


        ##search for P50 between 0 and 70ms:

        arg70ms = int(np.round(batch.size(dim=0) /total_steps_ms *70 ))
        p50 = torch.max(batch[0:arg70ms])

        arg_p50 = torch.argmax(batch[0:arg70ms])
        p50_moment1 = torch.tensor(moment(batch[0:arg70ms], moment=1) , dtype=torch.float32)  #mean
        #p50_moment2 = torch.tensor(moment(x[0:arg70ms], moment=2) )   #variance
        #p50_moment3 = torch.tensor(moment(x[0:arg70ms], moment=3) )  #skewness
        #p50_moment4 = torch.tensor(moment(x[0:arg70ms], moment=4) )  #kurtosis

     
        
        sum_stats_vec = torch.stack([p50, arg_p50, p50_moment1, 
     ])


        batch_list.append(sum_stats_vec)

    sum_stats = torch.stack(batch_list)

    print('sum stats', type(sum_stats))
    print(sum_stats)

    
    return sum_stats



def calculate_summary_stats_N100(x):
    """
    Input: observations or simulations
    Returns summary statistics 
    specific for event related potentials
    extracts the following:
     N100: time, value, mean, variance, skewness and kurtosis of the time interval around it

    """
  
    time_window = 30

    batch_list = []

    for batch in x:
     
        total_steps_ms = batch.size(dim=0)/time_window

        ## search for N100 
        arg200ms = int(np.round(batch.size(dim=0) /total_steps_ms *200))


        N100 = torch.min(batch[:arg200ms])

        arg_N100 = torch.argmin(batch[:arg200ms])
        N100_moment1 = torch.tensor(moment(batch[0:arg200ms], moment=1), dtype=torch.float32)   #mean
        #N100_moment2 = torch.tensor(moment(x[0:arg200ms], moment=2))    #variance
        #N100_moment3 = torch.tensor(moment(x[0:arg200ms], moment=3))    #skewness
        #N100_moment4 = torch.tensor(moment(x[0:arg200ms], moment=4))    #kurtosis


        sum_stats_vec = torch.stack([N100,
        arg_N100, 
        N100_moment1, #N100_moment2, N100_moment3, #N100_moment4,
                ])


        batch_list.append(sum_stats_vec)

    sum_stats = torch.stack(batch_list)

    
    return sum_stats

def calculate_summary_stats_P200(x):
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
        total_steps_ms = batch.size(dim=0)/time_window

        #sets the first value as baseline
        batch = torch.sub(batch, torch.index_select(batch,0,torch.tensor([0])))


        ##search for P50 between 0 and 70ms:

        arg70ms = int(np.round(batch.size(dim=0) /total_steps_ms *70 ))
 
        P200 = torch.max(batch[arg70ms:])

        arg_P200 = torch.argmax(batch[arg70ms:])
        P200_moment1 = torch.tensor(moment(batch[arg70ms:], moment=1), dtype=torch.float32)   #mean
        #P200_moment2 = torch.tensor(moment(x[arg70ms:], moment=2))    #variance
        #P200_moment3 = torch.tensor(moment(x[arg70ms:], moment=3))   #skewness
        #P200_moment4 = torch.tensor(moment(x[arg70ms:], moment=4))    #kurtosis



        
        sum_stats_vec = torch.stack([P200, 
        arg_P200,
        P200_moment1, #P200_moment2, P200_moment3, #P200_moment4
                ])



        batch_list.append(sum_stats_vec)

    sum_stats = torch.stack(batch_list)


    
    return sum_stats


def calculate_summary_statistics_alternative(x):
    x = x.unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(input=x, size=100)
    x = x.squeeze(0).squeeze(0)
    sum_stat = torch.sub(x, torch.index_select(x,0,torch.tensor([0])))

    return sum_stat
