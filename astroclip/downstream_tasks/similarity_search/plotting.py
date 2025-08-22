import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_similar_images(
    model_names: tuple,
    query_images: list,
    sims: dict,
    similarity_type: str = "im_sim",
    num_retrievals: int = 8,
    save_dir: str = None,
    similarity_scores: list = None,
):
    """Functionality for plotting retrieved galaxy images"""
    plt.figure(figsize=[19.4, 6.1])
    for n, img in enumerate(query_images):
        plt.subplot(len(query_images), 13, n * 13 + 1)
        plt.imshow(img.T)
        plt.axis("off")
        for j in range(num_retrievals):
            plt.subplot(len(query_images), 13, n * 13 + j + 1 + 1)
            plt.imshow(sims[n][similarity_type][j].T)
            if similarity_scores is not None and len(similarity_scores) > n:
                score = similarity_scores[n][j]
                plt.title(f"{score:.2f}")
            plt.axis("off")
    plt.subplots_adjust(wspace=0.01, hspace=0.0)
    plt.subplots_adjust(wspace=0.00, hspace=0.01)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"retrieval_{model_names[0]},{model_names[1]}_{similarity_type}.png"))
    else:
        plt.savefig(os.path.join(save_dir, f"retrieval_{model_names[0]},{model_names[1]}_{similarity_type}.png"))


def plot_similar_spectra(
    model_names: tuple,
    query_spectra: list,
    query_images: list,
    sims: dict,
    similarity_type: str = "im_sim",
    num_retrievals: int = 5,
    save_dir: str = None,
    similarity_scores: list = None,
):
    """Functionality for plotting retrieved galaxy spectra"""
    l = np.linspace(3586.7408577, 10372.89543574, query_spectra[0].shape[0])
    figure = plt.figure(figsize=[15, 5])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for n, sp in enumerate(query_spectra):
        plt.subplot(1, len(query_spectra), n + 1)
        plt.plot(
            l,
            gaussian_filter1d(sp[:, 0], 5),
            color=colors[n],
            lw=1,
            label="spectrum of query image",
        )

        for j in range(num_retrievals):
            score = 0
            if similarity_scores is not None and len(similarity_scores) > n:
                score = similarity_scores[n][j+1]
            if j == 0:
                plt.plot(
                    l,
                    gaussian_filter1d(sims[n][similarity_type][j][:, 0], 5),
                    alpha=0.5,
                    lw=1,
                    color="gray",
                    label="retrieved spectra",
                )
            else:
                plt.plot(
                    l,
                    gaussian_filter1d(sims[n][similarity_type][j][:, 0], 5),
                    alpha=0.5,
                    lw=1,
                    color="gray",
                )
            # set y lim
            plt.ylim(1.1 * min(sp[:, 0]), 1.1 * max(sp[:, 0]))
        plt.title(f"Cosine similarity: {score:.2f}")
        plt.xlabel(r"$\lambda$", fontsize=18)
        plt.ylabel("flux", fontsize=18)
        plt.legend(fontsize=18, loc="lower right")

        # Add inset image to the first subplot
        axins = plt.gca().inset_axes([0, 0.55, 0.4, 0.4])
        image_data = query_images[n]
        axins.imshow(image_data.T)
        axins.axis("off")

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"retrieval_{model_names[0]},{model_names[1]}_{similarity_type}.png"))
    else:
        plt.savefig(os.path.join(save_dir, f"retrieval_{model_names[0]},{model_names[1]}_{similarity_type}.png"))

