Controlling summary stats after having induced stochasticity into the code:

![](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/notebooks/summary_stats1.png)

![summary_stats2](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/notebooks/summary_stats2.png)



Thetas of samples: 

```
tensor([[ 76.0188, 135.6603],
        [ 73.5117, 135.0829],
        [ 78.7453, 133.4204],
        [ 77.0498, 137.5991],
        [ 72.2006, 137.0394],
        [ 73.9815, 142.5612],
        [ 71.9782, 137.9698],
        [ 78.6449, 137.5173],
        [ 72.4756, 133.3963],
        [ 79.4789, 130.8557]])
```

Summary statistics: `torch.stack([p50, N100, P200, arg_p50, arg_N100, arg_P200,`

​    `p50_moment1, p50_moment2, p50_moment3, p50_moment4,`

​    `N100_moment1, N100_moment2, N100_moment3, N100_moment4,`

​    `P200_moment1,P200_moment2, P200_moment3, P200_moment4]`

![image-20211105084724889](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/hist_100_samples.png)



Toy example with the first distal drive and the second proximal drive as parameters. 1000 simulations, 500 samples drawn from posterior.

Meta-data can be found here: sbi_for_eeg_data/code/results/ERP_nsf11-08-2021_00:05:33

![](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/code/results/ERP_nsf11-08-2021_00:05:33/figure4.png)

![figure5](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/code/results/ERP_nsf11-08-2021_00:05:33/figure5.png)


