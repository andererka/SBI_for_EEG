import torch
from torch import Tensor
from typing import Tuple, List, Optional, Dict
from matplotlib.pyplot import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

import matplotlib.cm as cm
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable


from scipy import spatial

import cycler



import collections
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import six
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import binom, gaussian_kde
from torch import Tensor

from sbi.utils import eval_conditional_density



try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


# taken from mackelab/Identifying-informative-features-of-HH-models-using-SBI

color_theme = plt.cm.Greens(torch.linspace(0, 1, 2))
color_theme[1] = torch.tensor([225, 0, 0, 255]).numpy() / 255 
color_theme[0] = torch.tensor([31, 119, 180, 255]).numpy() / 255  

mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", color_theme)
mpl.rc("image", cmap="Blues")

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
    
    fact = 1.0 / (X.size(1) - 1)
    X -= torch.mean(X, dim=1, keepdim=True)
    Xt = X.t()  
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
   
    if batchsize > 0:
        # TODO: IF BASESAMPLE ALSO HAS BATCHES -> split and align them!
        batched_samples = samples
        Vars = []
        for i, samples in enumerate(batched_samples):
            rel_Var = compare_vars(list(samples), base_sample)
            Vars.append(rel_Var)

        Vars = torch.stack(Vars)
        if "mean" in agg_with.lower():
            Vars_agg = Vars.mean(0)
        if "median" in agg_with.lower():
            Vars_agg = Vars.median(0)[0]
        # Vars_std = Vars.std(0)
    else:
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


def conditional_pairplot_comparison(
    density: Any,
    density2: Any,
    condition: torch.Tensor,
    condition2: torch.Tensor,
    limits: Union[List, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    subset: List[int] = None,
    resolution: int = 50,
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    warn_about_deprecation: bool = True,
    fig=None,
    axes=None,
    color_map=None,
    alpha1 = 1,
    alpha2 = 1,
    color_contours = ['red', 'blue'],

    **kwargs,
):
    r"""
    Plot conditional distribution given all other parameters.
    The conditionals can be interpreted as slices through the `density` at a location
    given by `condition`.
    For example:
    Say we have a 3D density with parameters $\theta_0$, $\theta_1$, $\theta_2$ and
    a condition $c$ passed by the user in the `condition` argument.
    For the plot of $\theta_0$ on the diagonal, this will plot the conditional
    $p(\theta_0 | \theta_1=c[1], \theta_2=c[2])$. For the upper
    diagonal of $\theta_1$ and $\theta_2$, it will plot
    $p(\theta_1, \theta_2 | \theta_0=c[0])$. All other diagonals and upper-diagonals
    are built in the corresponding way.
    Args:
        density: Probability density with a `log_prob()` method.
        condition: Condition that all but the one/two regarded parameters are fixed to.
            The condition should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution.
        limits: Limits in between which each parameter will be evaluated.
        points: Additional points to scatter.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on)
        resolution: Resolution of the grid at which we evaluate the `pdf`.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.
        warn_about_deprecation: With sbi v0.15.0, we depracated the import of this
            function from `sbi.utils`. Instead, it should be imported from
            `sbi.analysis`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.
    Returns: figure and axis of posterior distribution plot
    """
    device = density._device if hasattr(density, "_device") else "cpu"

    device2 = density2._device if hasattr(density2, "_device") else "cpu"

    # Setting these is required because _pairplot_scaffold will check if opts['diag'] is
    # `None`. This would break if opts has no key 'diag'. Same for 'upper'.

    print('check')
    diag = "cond"
    upper = "cond"

    opts = _get_default_opts()


    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    opts['density'] = density
    opts['condition'] = condition

 
    opts2 = _get_default_opts()


    opts2 = _update(opts2, locals())
    opts2 = _update(opts2, kwargs)

 
    opts2['density'] = density2
    opts2['condition'] = condition2


    if color_map == None:
        opts["samples_colors"] = ['viridis', 'plasma']
    else:
        opts["samples_colors"] = color_map 

    opts['samples_colors2'] = color_contours



    dim, limits, eps_margins = prepare_for_conditional_plot(condition, opts)

    #dim, limits2, eps_margins2 = prepare_for_conditional_plot(condition2, opts2)

    #diag_func = get_conditional_diag_func(opts, opts2, limits, eps_margins,  resolution)


    def diag_func(row, **kwargs):
        p_vector = (
            eval_conditional_density(
                opts["density"],
                opts["condition"],
                limits,
                row,
                row,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[row],
                warn_about_deprecation=False,
            )
            .to("cpu")
            .numpy()
        )
        p_vector2 = (
            eval_conditional_density(
                opts2["density"],
                opts2["condition"],
                limits,
                row,
                row,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[row],
                warn_about_deprecation=False,
            )
            .to("cpu")
            .numpy()
        )
        h = plt.plot(
            np.linspace(
                limits[row, 0],
                limits[row, 1],
                resolution,
            ),
            p_vector,
            c=opts['samples_colors2'][0],
        )
        h2 = plt.plot(
            np.linspace(
                limits[row, 0],
                limits[row, 1],
                resolution,
            ),
            p_vector2,
            c=opts['samples_colors2'][1],
        )

    opts['lower'] = None


    def upper_func(row, col, **kwargs):
        p_image = (
            eval_conditional_density(
                opts["density"],
                opts["condition"].to(device),
                limits.to(device),
                row,
                col,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[col],
                warn_about_deprecation=False,
            )
            .to("cpu")
            .numpy()
        )
        p_image2 = (
            eval_conditional_density(
                opts2["density"],
                opts2["condition"].to(device2),
                limits.to(device2),
                row,
                col,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[col],
                warn_about_deprecation=False,
                
            )
            .to("cpu")
            .numpy()
        )


        h = plt.imshow(
            p_image.T,
            origin="lower",
            extent=[
                limits[col, 0],
                limits[col, 1],
                limits[row, 0],
                limits[row, 1],
            ],
            aspect="auto",
            cmap=opts["samples_colors"][0],
            alpha = alpha1

        )

        h2 = plt.imshow(
            p_image2.T,
            origin="lower",
            extent=[
                limits[col, 0],
                limits[col, 1],
                limits[row, 0],
                limits[row, 1],
                
            ],
            aspect="auto",
            cmap=opts["samples_colors"][1],
            alpha=alpha2
        )





    return _arrange_plots(
        diag_func, upper_func, dim, limits,  points, opts,  fig=fig, axes=axes
    )



def _arrange_plots(
    diag_func, upper_func, dim, limits, points, opts, fig=None, axes=None
):
    """
    Arranges the plots for any function that plots parameters either in a row of 1D
    marginals or a pairplot setting.
    Args:
        diag_func: Plotting function that will be executed for the diagonal elements of
            the plot (or the columns of a row of 1D marginals). It will be passed the
            current `row` (i.e. which parameter that is to be plotted) and the `limits`
            for all dimensions.
        upper_func: Plotting function that will be executed for the upper-diagonal
            elements of the plot. It will be passed the current `row` and `col` (i.e.
            which parameters are to be plotted and the `limits` for all dimensions. None
            if we are in a 1D setting.
        dim: The dimensionality of the density.
        limits: Limits for each parameter.
        points: Additional points to be scatter-plotted.
        opts: Dictionary built by the functions that call `_arrange_plots`. Must
            contain at least `labels`, `subset`, `figsize`, `subplots`,
            `fig_subplots_adjust`, `title`, `title_format`, ..
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
    Returns: figure and axis
    """

    # Prepare points
    if points is None:
        points = []
    if type(points) != list:
        points = ensure_numpy(points)
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]

    # TODO: add asserts checking compatibility of dimensions

    # Prepare labels
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Figure out if we subset the plot
    subset = opts["subset"]
    if subset is None:
        rows = cols = dim
        subset = [i for i in range(dim)]
    else:
        if type(subset) == int:
            subset = [subset]
        elif type(subset) == list:
            pass
        else:
            raise NotImplementedError
        rows = cols = len(subset)
    flat = upper_func is None
    if flat:
        rows = 1
        opts["lower"] = None

    # Create fig and axes if they were not passed.
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            rows, cols, figsize=opts["figsize"], **opts["subplots"]
        )
    else:
        assert axes.shape == (
            rows,
            cols,
        ), f"Passed axes must match subplot shape: {rows, cols}."
    # Cast to ndarray in case of 1D subplots.
    axes = np.array(axes).reshape(rows, cols)

    # Style figure
    fig.subplots_adjust(**opts["fig_subplots_adjust"])
    fig.suptitle(opts["title"], **opts["title_format"])

    # Style axes
    row_idx = -1
    for row in range(rows):
        if row not in subset and not flat:
            continue
        else:
            row_idx += 1

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            if flat:
                current = "diag"
            elif row == col:
                current = "diag"
            elif row < col:
                current = "upper"
            else:
                current = "lower"

            ax = axes[row_idx, col_idx]
            plt.sca(ax)

            # Background color
            if (
                current in opts["fig_bg_colors"]
                and opts["fig_bg_colors"][current] is not None
            ):
                ax.set_facecolor(opts["fig_bg_colors"][current])

            # Axes
            if opts[current] is None:
                ax.axis("off")
                continue

            # Limits
            ax.set_xlim((limits[col][0], limits[col][1]))
            if current != "diag":
                ax.set_ylim((limits[row][0], limits[row][1]))

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_position(("outward", opts["despine"]["offset"]))

            # Formatting axes
            if current == "diag":  # off-diagnoals
                if opts["lower"] is None or col == dim - 1 or flat:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            else:  # off-diagnoals
                if row == dim - 1:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            if opts["tick_labels"] is not None:
                ax.set_xticklabels(
                    (
                        str(opts["tick_labels"][col][0]),
                        str(opts["tick_labels"][col][1]),
                    )
                )

            # Diagonals       diag_func2(row=col, limits=limits)

            if current == "diag":
              
                diag_func(row=col, limits=limits)

                if len(points) > 0:
                    extent = ax.get_ylim()
                    for n, v in enumerate(points):
                        h = plt.plot(
                            [v[:, col], v[:, col]],
                            extent,
                            color=opts["points_colors"][n],
                            **opts["points_diag"],
                        )

            # Off-diagonals
            else:
           
                upper_func(
                    row=row,
                    col=col

                )

                if len(points) > 0:

                    for n, v in enumerate(points):
                        h = plt.plot(
                            v[:, col],
                            v[:, row],
                            color=opts["points_colors"][n],
                            **opts["points_offdiag"],
                        )

    if len(subset) < dim:
        if flat:
            ax = axes[0, len(subset) - 1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
        else:
            for row in range(len(subset)):
                ax = axes[row, len(subset) - 1]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
                ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
                if row == len(subset) - 1:
                    ax.text(
                        x1 + (x1 - x0) / 12.0,
                        y0 - (y1 - y0) / 1.5,
                        "...",
                        rotation=-45,
                        **text_kwargs,
                    )

    return fig, axes



def ensure_numpy(t: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Returns np.ndarray if torch.Tensor was provided.
    Used because samples_nd() can only handle np.ndarray.
    """
    if isinstance(t, torch.Tensor):
        return t.numpy()
    else:
        return t


def prepare_for_conditional_plot(condition, opts):
    """
    Ensures correct formatting for limits. Returns the margins just inside
    the domain boundaries, and the dimension of the samples.
    """
    # Dimensions
    dim = condition.shape[-1]

    # Prepare limits
    if len(opts["limits"]) == 1:
        limits = [opts["limits"][0] for _ in range(dim)]
    else:
        limits = opts["limits"]
    limits = torch.as_tensor(limits)

    # Infer the margin. This is to avoid that we evaluate the posterior **exactly**
    # at the boundary.
    limits_diffs = limits[:, 1] - limits[:, 0]
    eps_margins = limits_diffs / 1e5

    return dim, limits, eps_margins


def _format_axis(ax, xhide=True, yhide=True, xlabel="", ylabel="", tickformatter=None):
    for loc in ["right", "top", "left", "bottom"]:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel("")
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel("")
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines["bottom"].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax


def _get_default_opts():
    """Return default values for plotting specs."""

    return {
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # ticks
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 50, "density": False, "histtype": "step"},
        "hist_offdiag": {
            # 'edgecolor': 'none',
            # 'linewidth': 0.0,
            "bins": 50,
        },
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 50, "color": "black"},
        "kde_offdiag": {"bw_method": "scott", "bins": 50},
        # options for contour
        "contour_offdiag": {"levels": [0.68], "percentile": True},
        # options for scatter
        "scatter_offdiag": {
            "alpha": 0.5,
            "edgecolor": "none",
            "rasterized": False,
        },
        "scatter_diag": {},
        # options for plot
        "plot_offdiag": {},
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {
            "marker": ".",
            "markersize": 20,
        },
        # other options
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
    }



def get_conditional_diag_func(opts, opts2, limits, eps_margins, resolution):
    """
    Returns the diag_func which returns the 1D marginal conditional plot for
    the parameter indexed by row.
    """

    def diag_func(row, **kwargs):
        p_vector = (
            eval_conditional_density(
                opts["density"],
                opts["condition"],
                limits,
                row,
                row,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[row],
                warn_about_deprecation=False,
            )
            .to("cpu")
            .numpy()
        )
        p_vector2 = (
            eval_conditional_density(
                opts2["density"],
                opts2["condition"],
                limits,
                row,
                row,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[row],
                warn_about_deprecation=False,
            )
            .to("cpu")
            .numpy()
        )
        h = plt.plot(
            np.linspace(
                limits[row, 0],
                limits[row, 1],
                resolution,
            ),
            p_vector,
            c='red',
        )
        h2 = plt.plot(
            #np.linspace(
            ##    limits[row, 0],
            #    limits[row, 1],
            #    resolution,
            #),
            p_vector2,
            c='blue',
        )
        

    return diag_func



def _update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = _update(dv, v)
        else:
            d[k] = v
    return d



def pairplot_comparison(
    samples: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ] = None,
    samples2: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: List[int] = None,
    upper: Optional[str] = "hist",
    diag: Optional[str] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    color_map = ['viridis', 'Greys'],

    **kwargs,
):
    """
    Plot samples in a 2D grid showing marginals and pairwise marginals.
    Each of the diagonal plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from. Each upper-diagonal plot can be interpreted as a
    2D-marginal of the distribution.
    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        upper: Plotting style for upper diagonal, {hist, scatter, contour, cond, None}.
        diag: Plotting style for diagonal, {hist, cond, None}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.
    Returns: figure and axis of posterior distribution plot
    """

    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: add legend (if legend is True)

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)
    samples2, dim, limits = prepare_for_plot(samples2, limits)

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    #diag_func = get_diag_func2(samples, samples2, limits, opts, **kwargs)

    def diag_func(row, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["diag"][n] == "hist":
                    h = plt.hist(
                        v[:, row], color='red', **opts["hist_diag"]
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    h = plt.plot(
                        xs,
                        ys,
                        color='red',
                    )
                elif "upper" in opts.keys() and opts["upper"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color='red',
                            **opts["scatter_diag"],
                        )
                else:
                    pass

            for n, v in enumerate(samples2):
                if opts["diag"][n] == "hist":
                    h = plt.hist(
                        v[:, row], color='blue', **opts["hist_diag"]
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    h = plt.plot(
                        xs,
                        ys,
                        color='blue',
                    )
                elif "upper" in opts.keys() and opts["upper"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color='blue',
                            **opts["scatter_diag"],
                        )
                else:
                    pass


    def upper_func(row, col, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["upper"][n] == "hist" or opts["upper"][n] == "hist2d":
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"],
                    )
                    h = plt.imshow(
                        hist.T,
                        opts["hist_offdiag"][0],
                        origin="lower",
                        extent=[
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ],
                        aspect="auto",
                        cmap=color_map[0],
                        
                    )

                if "contour" in opts["upper"][n] or "kde" in opts["upper"][n] or "contourf" in opts["upper"][n] or "kde2d" in opts["upper"][n]:
  
                    density = gaussian_kde(
                        v[:, [col, row]].T,
                        bw_method=opts["kde_offdiag"][0]["bw_method"],
                    )
                    X, Y = np.meshgrid(
                        np.linspace(
                            limits[col][0],
                            limits[col][1],
                            opts["kde_offdiag"][0]["bins"],
                        ),
                        np.linspace(
                            limits[row][0],
                            limits[row][1],
                            opts["kde_offdiag"][0]["bins"],
                        ),
                    )
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(density(positions).T, X.shape)

                    if "kde" in opts["upper"][n] or "kde2d" in opts["upper"][n]:
                        h = plt.imshow(
                            Z,
                            cmap = opts["kde_offdiag"][0]['cmap'],
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            origin="lower",
                            aspect="auto",
                          
                        )
                    if "contour" in opts["upper"][n]:
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        h = plt.contour(
                            X,
                            Y,
                            Z,
                            origin="lower",
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            colors='red',
                            levels=opts["contour_offdiag"]["levels"],
                        )
                    else:
                        pass
                if "scatter" in opts["upper"][n] :
                    h = plt.scatter(
                        v[:, col],
                        v[:, row],
                        color='red',
                        **opts["scatter_offdiag"][0],
                    )
                if "plot" in opts["upper"][n] :
                    h = plt.plot(
                        v[:, col],
                        v[:, row],
                        color='red',
                        **opts["plot_offdiag"][0],
                    )
                else:
                    pass

            for n, v in enumerate(samples2): 
                if "hist" in opts["upper"][n] or "hist2d" in opts["upper"][n]:
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"][1],
                    )
                    h2 = plt.imshow(
                        hist.T,
                        opts["hist_offdiag"][1],
                        origin="lower",
                        extent=[
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ],
                        aspect="auto",
                        cmap=color_map[1],
                        
                    )

                if "contour" in opts["upper"][n] or "kde" in opts["upper"][n] or "contourf" in opts["upper"][n] or "kde2d" in opts["upper"][n]:
                    
                    density = gaussian_kde(
                        v[:, [col, row]].T,
                        bw_method=opts["kde_offdiag"][1]["bw_method"],
                    )
                    X, Y = np.meshgrid(
                        np.linspace(
                            limits[col][0],
                            limits[col][1],
                            opts["kde_offdiag"][1]["bins"],
                        ),
                        np.linspace(
                            limits[row][0],
                            limits[row][1],
                            opts["kde_offdiag"][1]["bins"],
                        ),
                    )
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(density(positions).T, X.shape)

                    if "kde" in opts["upper"][n] or "kde2d" in opts["upper"][n]:
                        h2 = plt.imshow(
                            Z,
                            cmap = opts["kde_offdiag"][1]['cmap'],
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            
                            origin="lower",
                            aspect="auto",
                            
                        )
                    if "contour" in opts["upper"][n]:
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        h2 = plt.contour(
                            X,
                            Y,
                            Z,
                            origin="lower",
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            colors='blue',
                            levels=opts["contour_offdiag"]["levels"],
                        )
                    else:
                        pass
                elif opts["upper"][n] == "scatter":
                    h2 = plt.scatter(
                        v[:, col],
                        v[:, row],
                        color='blue',
                        **opts["scatter_offdiag"][1],
                    )
                elif opts["upper"][n] == "plot":
                    h2 = plt.plot(
                        v[:, col],
                        v[:, row],
                        color='blue',
                        **opts["plot_offdiag"][1],
                    )
                else:
                    pass

    return _arrange_plots(
        diag_func, upper_func, dim, limits, points, opts, fig=fig, axes=axes
    )


def get_diag_func2(samples, samples2, limits, opts, **kwargs):
    """
    Returns the diag_func which returns the 1D marginal plot for the parameter
    indexed by row.
    """

    def diag_func(row, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["diag"][n] == "hist":
                    h = plt.hist(
                        v[:, row], color=opts["samples_colors"][n], **opts["hist_diag"]
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    h = plt.plot(
                        xs,
                        ys,
                        color='red',
                    )
                elif "upper" in opts.keys() and opts["upper"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color=opts["samples_colors"][n],
                            **opts["scatter_diag"],
                        )
                else:
                    pass

            for n, v in enumerate(samples2):
                if opts["diag"][n] == "hist":
                    h = plt.hist(
                        v[:, row], color=opts["samples_colors"][n], **opts["hist_diag"]
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    h = plt.plot(
                        xs,
                        ys,
                        color='blue',
                    )
                elif "upper" in opts.keys() and opts["upper"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color=opts["samples_colors"][n],
                            **opts["scatter_diag"],
                        )
                else:
                    pass

    return diag_func


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified
    percentile levels.
    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]. Specifies contour levels that
        include a given proportion of samples, i.e., 0.1 specifies where the top 10% of
        the density is.
    Return
    ------
    Array of same shape as probs with percentile labels. Values in output array
    denote labels which percentile bin the probability mass belongs to.
    Example: for levels = [0.1, 0.5], output array will take on values [1.0, 0.5, 0.1],
    where elements labeled "0.1" correspond to the top 10% of the density, "0.5"
    corresponds to between top 50% to 10%, etc.
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def prepare_for_plot(samples, limits):
    """
    Ensures correct formatting for samples and limits, and returns dimension
    of the samples.
    """

    # Prepare samples
    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    # Prepare limits. Infer them from samples if they had not been passed.
    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)
    return samples, dim, limits

