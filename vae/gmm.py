import sklearn.mixture
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib.patheffects as mpatheffects


def get_gmm_and_pos_label(array, n_components=2, n_steps=5000):
    gmm = sklearn.mixture.GaussianMixture(
        n_components=n_components, covariance_type="spherical", random_state=0
    )
    gmm.fit(array.reshape(-1, 1))
    label = np.argmax(gmm.means_)

    # low = array.min()
    # high = array.max()

    low = gmm.means_.min() - 2 * np.sqrt(gmm.covariances_[np.argmin(gmm.means_)])
    high = gmm.means_.max() + 2 * np.sqrt(gmm.covariances_[np.argmax(gmm.means_)])

    ref_space = np.linspace(low, high, n_steps)
    result = gmm.predict(ref_space.reshape(-1, 1))

    idx = np.where(np.ediff1d(result) != 0)
    cutoffs = ref_space[idx]

    return gmm, label, cutoffs


def plot_gmm_fitting(array, gmm, ax):
    plt.sca(ax)
    _ = plt.hist(array.flatten(), color="lightgray", bins=200, density=True)
    x = np.linspace(array.min(), array.max(), 200)

    log_prob = gmm.score_samples(x.reshape(-1, 1))
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(log_prob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    mean_index = np.argmax(pdf_individual, axis=0)
    rank_map = mean_index.argsort().argsort()

    ax.set_prop_cycle(color=plt.get_cmap("Dark2")(rank_map))
    ax.plot(x, pdf_individual)
    ax.plot(x, pdf, "--k")
    return ax


def auto_gate_func(array, n_components=3, n_stds=3, log_transform=True):
    gmm = sklearn.mixture.GaussianMixture(
        n_components=n_components, covariance_type="spherical", random_state=0
    )
    if log_transform:
        gmm.fit(np.log1p(array).reshape(-1, 1))
    else:
        gmm.fit(array.reshape(-1, 1))
    means = gmm.means_
    stds = np.sqrt(gmm.covariances_)
    idx = np.argmax(means)
    lower_bound = means[idx] - n_stds * stds[idx]
    if log_transform:
        return np.expm1(lower_bound)
    else:
        return lower_bound


def plot_cumulative(array, ax, hist_kwargs={}):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    _ = ax.hist(array, histtype="step", bins=300, cumulative=1, **hist_kwargs)

    return ax


def gmm_label_map_by_mean(gmm):
    return {
        o: n
        for o, n in zip(
            range(len(gmm.means_)),
            sorted(range(len(gmm.means_)), key=lambda x: gmm.means_[x][0]),
        )
    }


def sort_predict_label(gmm, labels):
    mapping = gmm_label_map_by_mean(gmm)
    sorted_labels = labels.copy()
    for k, v in mapping.iteritems():
        sorted_labels[labels == k] = v
    return sorted_labels


def plot_hist_gmm(
    df,
    markers,
    n_components=2,
    subplot_grid_shape=None,
    transform_log=True,
    xlim_percentiles=(0, 100),
    cum_density=False,
    hide_yaxis_left=True,
):
    if transform_log:
        df = df.transform(np.log1p)
        revert_func = np.expm1
    else:
        revert_func = np.array
    if subplot_grid_shape is None:
        subplot_grid_shape = (1, len(markers))
    n_rows, n_cols = subplot_grid_shape
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True)
    axes = np.array(axes)

    for m, ax in zip(markers, axes.ravel()):
        gmm, _, cutoffs = get_gmm_and_pos_label(df[m].values, n_components=n_components)
        plot_gmm_fitting(df[m].values, gmm, ax)
        ax.title.set_text(m)
        if hide_yaxis_left:
            ax.yaxis.set_visible(False)

        p1, p2 = np.array(xlim_percentiles) / 100
        axis_min = df.loc[:, markers].quantile(p1).min()
        axis_max = df.loc[:, markers].quantile(p2).max()

        color_cum = "gray"

        pax = ax.twinx()
        pax = plot_cumulative(
            df[m].values, pax, hist_kwargs=dict(color=color_cum, density=cum_density)
        )
        pax.tick_params(axis="y", labelsize=8, colors=color_cum)

        print(cutoffs)

        cutoff_range = np.ptp(cutoffs)
        if cutoff_range == 0:
            cutoff_range = 1
        cutoff_colors = plt.get_cmap("plasma")(
            (cutoffs - np.min(cutoffs)) / cutoff_range
        )

        for co, cc in zip(cutoffs, cutoff_colors):
            ax.axvline(x=co, c=cc, alpha=0.2)
            ax.annotate(
                "",
                xy=(co, 0),
                xytext=(co, -0.05),
                xycoords=("data", "axes fraction"),
                arrowprops=dict(
                    arrowstyle="wedge, tail_width=0.7, shrink_factor=0.5", color=cc
                ),
            )
        ax.set_xlim(axis_min, axis_max)
        # cutoff_string = np.round(revert_func(cutoffs)).astype(int)

        for i, (co, cc) in enumerate(
            zip(revert_func(cutoffs)[::-1], cutoff_colors[::-1])
        ):
            text = ax.text(
                ax.get_xlim()[0] + 0.02 * np.diff(ax.get_xlim()),
                ax.get_ylim()[1] - 0.05 * (i + 1) * np.diff(ax.get_ylim()),
                f"{np.round(co).astype(int)}",
                fontsize=10,
                c=cc,
            )
            text_outline = mpatheffects.Stroke(linewidth=1, foreground="#000")
            text.set_path_effects([text_outline, mpatheffects.Normal()])
    plt.tight_layout()
    for aax in fig.axes:
        aax.spines["right"].set_color(color_cum)
        power_label = aax.yaxis.get_offset_text()
        power_label.set_visible(False)
        aax.annotate(
            power_label.get_text(),
            xy=(1.02, 1.01),
            xycoords="axes fraction",
            fontsize=10,
            color=color_cum,
        )
    plt.sca(ax)
