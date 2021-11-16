

![](figures/Summary_stats_post_prior.png)

- in which cases does the direct estimation of the posterior density fail? samples not at all in accordance with observation x and why? 
- Benchmark paper: evaluation metrics -> negative log probability of true parameters averaged over different (θo,xo), only feasible if inference is amortized
- looks like SNPE_C is default when I use SNPE? is it?

Dayan/Abott: 

**6.Conductances and Morphology:** 

-'Oversimplified models can, of course, give misleading results, but excessively detailed models can obscure interesting results beneath inessential and unconstrained
complexity.'

-parameters: are we interested in conductances? we can only infer parameters on which simulator bases assumptions on



**APT paper**: APT learns a mapping from data to the true posterior by maximizing the probability of simulation parameters under the proposal posterior.

- APT also scales to high-dim data: 10k-dimensional image data without summary statistics
- are we automatically considering ‘atomic’ proposals with APT?-> YES, default uses 10 atoms. This is very interesting for our case, as integrals are replaced by sums and the parameter values with the 'highest probability' wins.
- Atomic APT trains the posterior model through a series of multiple-choice problems, by minimizing the cross-entropy loss from supervised classiﬁcation
- specialized neural network architectures can be used to exploit known structure in the
  data, as we have shown using RNNs and CNNs for time series and image data



