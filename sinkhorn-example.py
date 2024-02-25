"""
File: sinkhorn-example.py
Author: Chuncheng Zhang
Date: 2024-02-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-25 ------------------------
# Requirements and constants
import numpy as np

import ot.plot
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

from ot.datasets import make_1D_gauss as gauss
import opensimplex


# %% ---- 2024-02-25 ------------------------
# Function and class
opensimplex.seed(np.random.randint(65536))


def gen_noise(x: float, y: float, z: float) -> float:
    return opensimplex.noise3(x=x, y=x, z=z) + 1


def normalize(s: np.ndarray) -> np.ndarray:
    return s / np.sum(s)


def gen_signal(n: int, x: float, y: float) -> np.ndarray:
    s = np.array([gen_noise(x, y, t) for t in np.linspace(0, 5, n)])
    return normalize(s)


def plot_P(s1, s2, P):
    plt.figure()
    plt.plot(s1, label="Signal 1")
    plt.plot(np.sum(P, axis=1), label="Signal 1 (r)")
    plt.plot(s2, label="Signal 2")
    plt.plot(np.sum(P, axis=0), label="Signal 2 (r)")
    plt.legend()
    plt.show()


# %%

n1 = 100
n2 = 120
s1 = gen_signal(n1, 10, 20)
s2 = gen_signal(n2, 20, 10)

fig = plt.figure()
plt.plot(s1, label="Signal 1")
plt.plot(s2, label="Signal 2")
plt.legend()
plt.show()

# %%

# Source and target distribution
X = np.arange(n1).reshape(-1, 1)
Y = np.arange(n2).reshape(-1, 1)

Q, R, g, log = ot.lowrank_sinkhorn(
    X,
    Y,
    s1,
    s2,
    rank=5,  # 10,
    init="random",
    gamma_init="rescale",
    rescale_cost=True,
    warn=False,
    log=True,
)
P = log["lazy_plan"][:]

ot.plot.plot1D_mat(s1, s2, P, "OT matrix Low rank")

plot_P(s1, s2, P)

# %% ---- 2024-02-25 ------------------------
# Play ground
# Compute cost matrix for sinkhorn OT
M = ot.dist(X, Y)
M = M / np.max(M)

# %%
# Solve sinkhorn with different regularizations using ot.solve
list_reg = [0.05, 0.005, 0.001]
list_P_Sin = []

for reg in list_reg:
    P = ot.solve(M, s1, s2, reg=reg, max_iter=2000, tol=1e-8).plan
    list_P_Sin.append(P)

# %%
# Solve low rank sinkhorn with different ranks using ot.solve_sample
list_rank = [3, 10, 50]
list_P_LR = []

for rank in list_rank:
    P = ot.solve_sample(X, Y, s1, s2, method="lowrank", rank=rank).plan
    P = P[:]
    list_P_LR.append(P)

# %% ---- 2024-02-25 ------------------------
# Pending
# Plot sinkhorn vs low rank sinkhorn
pl.figure(1, figsize=(10, 4))

pl.subplot(1, 3, 1)
pl.imshow(list_P_Sin[0], interpolation="nearest")
pl.axis("off")
pl.title("Sinkhorn (reg=0.05)")

pl.subplot(1, 3, 2)
pl.imshow(list_P_Sin[1], interpolation="nearest")
pl.axis("off")
pl.title("Sinkhorn (reg=0.005)")

pl.subplot(1, 3, 3)
pl.imshow(list_P_Sin[2], interpolation="nearest")
pl.axis("off")
pl.title("Sinkhorn (reg=0.001)")
pl.show()

plot_P(s1, s2, list_P_Sin[0])

# %%
pl.figure(2, figsize=(10, 4))

pl.subplot(1, 3, 1)
pl.imshow(list_P_LR[0], interpolation="nearest")
pl.axis("off")
pl.title("Low rank (rank=3)")

pl.subplot(1, 3, 2)
pl.imshow(list_P_LR[1], interpolation="nearest")
pl.axis("off")
pl.title("Low rank (rank=10)")

pl.subplot(1, 3, 3)
pl.imshow(list_P_LR[2], interpolation="nearest")
pl.axis("off")
pl.title("Low rank (rank=50)")

pl.tight_layout()
pl.show()

plot_P(s1, s2, list_P_LR[0])
# %% ---- 2024-02-25 ------------------------
# Pending
