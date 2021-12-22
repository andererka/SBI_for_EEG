### Week summary 21 Dez



How can we evaluate which summary stats are useful?

- posterior predictive checks
- histograms

1000 simulations, 9 summary stats:

<img src="/home/kathi/snap/typora/46/.config/Typora/typora-user-images/image-20211221183211529.png" alt="image-20211221183211529" style="zoom: 40%;" /><img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/post_dens_9stats_1000sims.png" alt="image-20211221183323559" style="zoom:40%;" />

1000 simulations, 12 summary stats:

<img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/12stats_1000sims.png" alt="image-20211221183441026" style="zoom:40%;" /><img src="/home/kathi/snap/typora/46/.config/Typora/typora-user-images/image-20211221183615666.png" alt="image-20211221183615666" style="zoom:40%;" />

- Fazit: more summary stats need more simulations probably?

- gerade dabei Histogramme zu untersuchen - habe aber noch keine Ergebnisse dazu.





Comparing step-wise and non-step-wise approach (both 900 simulations in total):

Sequential approach: 

<p float="left">
    <img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/sequential_1.png" style="zoom: 67%;" />
    <img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/sequential_combined_prior.png" style="zoom:67%;" />
</p>



-<img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/sequential_posterior_dens_plot.png" style="zoom: 100%;" />

Comparison to 'usual' approach:



<p float="left">
    <img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/non_sequential.png" style="zoom: 67%;"/>
    <img src="/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/non_sequential_prior.png" style="zoom: 67%;" />
</p>

-![non_squential_toy](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/non_squential_toy.png)





Weitere Ziele:



- Vergleich des HNN Modells mit anderem Modell?
- Summary statistics genauer untersuchen!
- more parameters for Incremental approach