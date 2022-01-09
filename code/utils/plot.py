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

from scipy import spatial

import matplotlib as mpl
import cycler


### following adapted from: 
# permalink: https://github.com/mackelab/Identifying-informative-features-of-HH-models-using-SBI/blob/9a4f8b9910d573923fb0cffcadb17363521050a9/code/sbi_feature_importance/analysis.py#L234

color_theme = plt.cm.Greens(torch.linspace(0, 1, 5))
color_theme[0] = torch.tensor([225, 0, 0, 255]).numpy() / 255  # ground truth / direct
color_theme[1] = torch.tensor([31, 119, 180, 255]).numpy() / 255  # post-hoc
color_theme[2] = torch.tensor([255, 127, 14, 255]).numpy() / 255  # full Orange
color_theme[0] = torch.tensor([44, 160, 44, 255]).numpy() / 255  # x_o
color_theme[3] = torch.tensor([23, 190, 207, 255]).numpy() / 255  # direct
mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", color_theme)
mpl.rc("image", cmap="Blues")


# taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI
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

# taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI
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

#taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI
def sample_KL(X: Tensor, Y: Tensor) -> float:
    """Uses nearest neighbour search to estimate the KL divergence of X and Y
    coming from 2 distributions P and Q.
    
    Args:
        X: Samples from P.
        Y: Samples from Q.
    Returns:
        kl: Estimate of the KL divergence.
    """
    n, d = X.shape
    m, d = Y.shape
    _, minXX = nearest_neighbours(X, X)
    _, minXY = nearest_neighbours(X, Y)

    kl1 = d / n * torch.sum(torch.log(minXY / minXX), dim=0)
    kl2 = torch.log(torch.tensor(m) / (torch.tensor(n) - 1))
    kl = kl1 + kl2
    return float(kl)


# taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI
def nearest_neighbours(X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes all nearest neighbour for all x_i in X given Y.
    
    Uses k-d trees to find nearest neighbours.
    
    Args:
        X: Sample for which to compute closest y_i for each x_i.
        Y: Sample used to look for nearest neighbours.
        
    Returns:
        Nearest neighbours in Y.
        Indices of nearest neighbours in Y.
    """
    tree = spatial.KDTree(Y)

    k = 2
    if torch.all(X != Y):
        k = 1

    def nn_Y(x):
        d, idx = tree.query(x, k=k, p=2)
        idx = torch.tensor(idx).view(-1, 1)
        d = torch.tensor(d).view(-1, 1)
        return torch.hstack([idx[k - 1], d[k - 1]])

    nns = [nn_Y(x) for x in X]
    nns = torch.vstack(nns)
    nn_idxs = list(nns[:, 0].int())
    return Y[nn_idxs], nns[:, 1]


# taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI
def compare_KLs(
    samples: List[Tensor], base_sample: Tensor, samplesize: int = 2000
) -> Tensor:
    """Computes estimate of the KL divergence between samples and samples from a
    reference distribution.
    
    Args:
        samples: List of samples to compare against a reference distribution. 
        base_sample: Samples from a reference distribution to compare the var
            against.
        sample_size: Number of samples to use in estimate of the KL.
    Returns:
        KLs: Estimates of the KL divergence for each sample in the list.
    """
    KLs = []
    for sample in samples:
        if sample.shape[0] < samplesize:
            samplesize = int(sample.shape[0])
        KL_i = sample_KL(sample[:samplesize], base_sample[:samplesize])
        KLs.append(torch.tensor(KL_i))

    KLs = torch.hstack(KLs)
    return KLs

# taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI
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


def plot_KLs(
    samples: List[Tensor],
    base_sample: Tensor,
    samplesize: int = 2000,
    plot_label: Optional[str] = None,
    idx: int = 0,
    kind: str = "bar",
    agg_with: str = "mean",
    batchsize=0,
) -> Axes:
    """Plot changes in variance per sample dim for a list of samples.
    
    Args:
        samples: List of samples to compare against a reference distribution. 
        base_sample: Samples from a reference distribution to compare the var
            against.
        sample_size: Number of samples to use in estimate of the KL.
        plot_label: Name of the figure.
        idx: idx, decides translation of bars in plot.
        kind: Which kind of plot to use.
        agg_with: How to aggregate the samples. Mean or Median are valid.
        batchsize: In case multiple arguments are supplied. Have to be in order!
            i.e. [[0,1,2],[0,1,2]] would supply two samples of batchsize 3.
    
    Returns:
        ax: plot axes.  
    """

    if batchsize > 0:
        # TODO: IF BASESAMPLE ALSO HAS BATCHES -> split and align them!
        KLs = []
        for i, samples in enumerate(samples):
            rel_KL = compare_KLs(list(samples), base_sample, samplesize)
            KLs.append(rel_KL)
        KLs = torch.vstack(KLs)
        if "mean" in agg_with.lower():
            KLs_agg = KLs.mean(0)
            KLs_disp = KLs.std(0)
        if "median" in agg_with.lower():
            KLs_agg = KLs.median(0)[0]
            KLs_disp_lower = KLs.median(0)[0] - KLs.quantile(0.25, 0)
            KLs_disp_upper = KLs.quantile(0.75, 0) - KLs.median(0)[0]
            KLs_disp = torch.vstack([KLs_disp_lower, KLs_disp_upper])
        if "box" in kind.lower():
            KLs_agg = KLs.T
    else:
        KLs_agg = compare_KLs(samples, base_sample, samplesize)
        KLs_disp = torch.zeros(2, len(KLs_agg))

    N = len(KLs_agg)
    if "bar" in kind.lower():
        ax = plt.bar(
            (torch.arange(N) + idx / (N + 2) - 1 / (N + 1)).numpy(),
            height=KLs_agg.numpy(),
            label=plot_label,
            align="edge",
            width=1 / (N + 2),
            yerr=KLs_disp.numpy(),
        )
    elif "points" in kind.lower():
        ax = plt.errorbar(
            range(N),
            KLs_agg.numpy(),
            yerr=KLs_disp,
            ls="",
            marker=".",
            label=plot_label,
        )
    elif "box" in kind.lower():
        if batchsize > 0:
            ax = plt.boxplot(
                KLs_agg.T.numpy(),
                positions=(torch.arange(N) + idx / (N) - 1 / (N + 1)).numpy(),
                widths=1 / (N),
                patch_artist=True,
                # boxprops={"color":color_theme[idx], "lw":1.5},
                # whiskerprops={"color":"black", "lw":1.5},
                # capprops={"color":color[idx], "lw":1.5},
                # flierprops={"markeredgecolor":color[idx]},
                medianprops={"color": "black"},
            )
            for patch in ax["boxes"]:
                patch.set_facecolor(color_theme[idx])
        else:
            ax = plt.plot(
                (torch.arange(N) + idx / (N + 2) - 1 / (N + 1)).numpy(),
                KLs_agg.numpy(),
                ls="",
                marker=">",
                c=color_theme[idx],
                mew=1,
                ms=6,
                label=plot_label,
            )
    plt.ylabel(r"$D_{KL}$")
    return ax
