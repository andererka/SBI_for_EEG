import torch
from torch import Tensor
from typing import Tuple, List, Optional, Dict
from matplotlib.pyplot import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

import matplotlib.cm as cm
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt


### following 3 functions adapted from: 
# permalink: https://github.com/mackelab/Identifying-informative-features-of-HH-models-using-SBI/blob/9a4f8b9910d573923fb0cffcadb17363521050a9/code/sbi_feature_importance/analysis.py#L234


def cov(X: Tensor, rowvar: bool = False):
    """Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        X: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `X` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    """
    X = X.clone()
    if X.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if X.dim() < 2:
        X = X.view(1, -1)
    if not rowvar and X.size(0) != 1:
        X = X.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (X.size(1) - 1)
    X -= torch.mean(X, dim=1, keepdim=True)
    Xt = X.t()  # if complex: mt = m.t().conj()
    return fact * X.matmul(Xt).squeeze()


def compare_vars(samples: List[Tensor], base_sample: Tensor) -> Tensor:
    """Computes ratio of sample variances vs variance of base sample for 
    each dimension.
    Args:
        samples: List of samples to compare against a reference distribution. 
        base_sample: samples from reference distribution 
    
    Returns:
        Ratios of variance for each parameter dimension and sample.
    """
    cov_base = cov(base_sample.clone())
    var_base = cov_base.diag()

    var_ratios = []
    for sample in samples:
        cov_sample = cov(sample.clone())
        var_sample = cov_sample.diag()

        var_ratios.append(var_sample / var_base)
    return torch.vstack(var_ratios).T


def plot_varchanges(
    samples: List[Tensor],
    base_sample: Tensor,
    yticklabels: Optional[str] = None,
    xticklabels: Optional[str] = None,
    zlims: Optional[Tuple[float, float]] = (None, None),
    agg_with: str = "mean",
    plot_label: str = None,
    batchsize=0,
) -> Axes:
    """Plot changes in variance per sample dim for a list of samples.
    
    Args:
        samples: List of samples to compare against a reference distribution. 
        base_sample: Samples from a reference distribution to compare the var
            against.
        yticklabels: Label for the yticks, i.e. parameter labels.
        xticklabels: Label for the xticks, i.e. summary features that are removed.
        zlims: Sets the limits for the colorbar.
        agg_with: How to aggregate the samples. Mean or Median are valid.
        add_cbar: Whether to add a colorbar to the plot
    Returns:
        ax: plot axes.
    """
   
    Vars_agg = compare_vars(samples, base_sample)

    ax = plt.gca()
    cmap = cm.get_cmap("coolwarm", 10)
    im = ax.imshow(
        Vars_agg.numpy(),
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=zlims[1])
    )
    ax.set_title(plot_label)
    if xticklabels != None:
        xrange = range(len(samples))
        plt.xticks(xrange, xticklabels)
        yrange = range(samples[0].shape[1])
        plt.yticks(yrange, yticklabels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    return im
