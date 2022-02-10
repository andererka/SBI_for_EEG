# Inferring neural microscale parameters from EEG-signals

My thesis project about extracting microscale parameters from the (macroscale) signal of an EEG or MEG with the help of simulation-based inference.

The simulations are based on the HNN-simulator: https://hnn.brown.edu/



**Neural Incremental Posterior Estimation:**

In order to infer a large parameter set from high dimensional data like a time series, we developed a new approach, that splits the inference process in separate parts. This can be done under the assumption that some parameters do not depend on some other parameters.

![](/home/kathi/Documents/Master_thesis/sbi_for_eeg_data/week_summaries/figures/scheme_model.png)





<u>Code structure:</u>

- **utils**
  - helpers.py: small helper functions, not specific
  - inference.py: functions that either only simulate theta and x, or infer posteriors, or both
  - plot.py: plotting functions, mostly adapted from the sbi toolbox
  - sbi_modulated_functions.py
  - simulation_wrapper: functions that take different number of parameters that are varied for simulations

- **summary_features**
  - calculate_summary_features.py: different functions to calculate a certain number of summary statistics. different functions can be compared against each other
- **data_load_writer**
  - load_from_file
  - write_to_file: defines a class that can store e.g. posteriors, priors, thetas, observations and meta data
