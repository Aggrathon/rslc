# This script is only inteded to use for benchmarking the RSLC algorithm. So the
# script is not inteded to be commonly used and, thus, the used libraries are
# not included in the requirements. However, the functions remain accessible,
# since the smileys might be a fun synthetic dataset for other clustering
# algorithms than RSLC.

from timeit import default_timer as timer
from typing import Tuple, List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import seaborn as sns

from rslc import cluster


def smiley_data(
    eye1: int = 100,
    eye2: int = 100,
    mouth: int = 300,
    outliers: int = 30,
    rnd: bool = True,
) -> np.ndarray:
    eye1 = np.random.normal(
        [np.repeat(3.0, eye1), np.linspace(-5.0, 6.0, eye1)], 0.8 * rnd
    )
    eye2 = np.random.normal(
        [np.repeat(-3.0, eye2), np.linspace(-5.0, 6.0, eye2)], 0.8 * rnd
    )
    angle = np.linspace(-np.pi - 0.3, 0.3, mouth)
    mouth = np.random.normal(
        [np.cos(angle) * 9.0, np.sin(angle) * 12.0 - 2.0], 0.8 * rnd
    )
    outliers = np.random.normal(
        [
            np.linspace(-12.0, 12.0, outliers),
            np.abs(np.linspace(-5.0, 5.0, outliers)) - 24.0,
        ],
        [np.repeat(0.5 * rnd, outliers), np.repeat(2.0 * rnd, outliers)],
    )
    return np.concatenate((eye1, eye2, mouth, outliers), -1).T


def categorical_from_int(array: List[int], labels: List[str]) -> pd.Series:
    cat = pd.Categorical(array)
    cat.categories = labels
    return cat


def smiley_truth(
    eye1: int = 100,
    eye2: int = 100,
    mouth: int = 300,
    outliers: int = 30,
    categorical=False,
) -> np.ndarray:
    clusters = np.concatenate(
        (
            np.repeat(1, eye1),
            np.repeat(2, eye2),
            np.repeat(0, mouth),
            np.repeat(3, outliers),
        )
    )
    if categorical:
        return categorical_from_int(clusters, ["Mouth", "Eye 1", "Eye 2", "Outliers"])
    return clusters


def smiley_n(n) -> Tuple[int, int, int, int]:
    return n * 3 // 15, n * 3 // 15, n * 8 // 15, n // 15


# Only run the benchmark if the script is called directly
if __name__ == "__main__":

    x = smiley_data()
    y = smiley_truth(categorical=True)
    sns.scatterplot(x[:, 0], x[:, 1], y)
    plt.axis("equal")
    plt.title("The Smiley Dataset")
    plt.show()

    dists = distance_matrix(x, x)
    clusters, outliers = cluster(dists, 4, 25)
    sns.scatterplot(
        x[:, 0],
        x[:, 1],
        categorical_from_int(clusters, ["Cluster " + str(i + 1) for i in range(4)]),
        categorical_from_int(outliers, ["Normal", "Outlier"]),
    )
    plt.axis("equal")
    plt.title("RSLC clustering and outlier detection")
    plt.show()

    np.random.seed(42)
    sizes = np.exp(np.linspace(np.log(100), np.log(10_000), 20))
    times = np.zeros_like(sizes)
    for i, s in enumerate(sizes):
        s = int(s)
        print("Clustering", s, "points")
        x = smiley_data(*smiley_n(s))
        d = distance_matrix(x, x)
        time = timer()
        clusters, outliers = cluster(d, 4, s // 10)
        time = timer() - time
        times[i] = time

    fig, ax = plt.subplots()
    sns.lineplot(sizes, times, ax=ax)
    for x, y in zip(sizes, times):
        ax.text(x - 300, y - 7, f"{y:.2f}", rotation=-40)
    plt.xlabel("Number of items")
    plt.ylabel("Seconds required by the clustering")
    plt.title("Time scaling of RSLC")
    plt.show()

    x = np.stack((np.ones_like(sizes), sizes, sizes ** 2, sizes ** 3, sizes ** 4), 1)
    model1, residuals1, _, _ = np.linalg.lstsq(x[:, :2], times)
    model2, residuals2, _, _ = np.linalg.lstsq(x[:, :3], times)
    model3, residuals3, _, _ = np.linalg.lstsq(x[:, :4], times)
    model4, residuals4, _, _ = np.linalg.lstsq(x, times)

    sns.barplot(
        np.arange(1, 5),
        np.concatenate([residuals1, residuals2, residuals3, residuals4]),
    )
    plt.title("Residuals from approximating the complexity")
    plt.show()

    x = np.linspace(100, 10_000, 100)
    x = np.stack((np.ones_like(x), x, x ** 2, x ** 3, x ** 4), 1)
    sns.lineplot(sizes, times, label="Measured times")
    sns.lineplot(x[:, 1], x[:, :2] @ model1, label="O(n)")
    sns.lineplot(x[:, 1], x[:, :3] @ model2, label="O(n^2)")
    sns.lineplot(x[:, 1], x[:, :4] @ model3, label="O(n^3)")
    sns.lineplot(x[:, 1], x @ model4, label="O(n^4)")
    plt.xlabel("Number of items")
    plt.ylabel("Seconds required by the clustering")
    plt.title("Approximating the complexity with polynomials")
    plt.show()
