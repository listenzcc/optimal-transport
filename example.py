"""
File: main.py
Author: Chuncheng Zhang
Date: 2024-02-23
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

# %% ---- 2024-02-23 ------------------------
# Requirements and constants
import ot
import ot.plot

import skdim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% ---- 2024-02-23 ------------------------
# Function and class
# generate data
data1, clusters = skdim.datasets.lineDiskBall(n=200, random_state=0)
data2 = skdim.datasets.swissRoll3Sph(n_swiss=400, n_sphere=200, h=2, random_state=0)

# plot
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "Scatter3d"}] * 2])

trace1 = go.Scatter3d(
    dict(zip(["x", "y", "z"], data1.T[:3])),
    mode="markers",
    name="data1",
    marker=dict(size=1.5, colorbar=dict()),
)
trace2 = go.Scatter3d(
    dict(zip(["x", "y", "z"], data2.T[:3])),
    mode="markers",
    name="data2",
    marker=dict(size=1.5, colorbar=dict()),
)

fig.add_traces([trace1, trace2], rows=1, cols=[1, 2])
fig.layout.update(height=450, width=600)
fig.show()

# %% ---- 2024-02-23 ------------------------
# Play ground
# data1.shape = (n1, 3), data2.shape = (n2, 4)

a = data1[:, :3]
b = data2[:, :3]
ua = np.ones((len(a),)) / len(a)
ub = np.ones((len(b),)) / len(b)

M = ot.dist(a, b)
im = plt.imshow(M)
plt.colorbar(im)
plt.show()

G0 = ot.emd(ua, ub, M)
im = plt.imshow(G0)
plt.colorbar(im)
plt.show()

# %%
plt.figure(4)
ot.plot.plot2D_samples_mat(a, b, G0, c=[0.5, 0.5, 1])
plt.plot(b[:, 0], b[:, 1], "+b", label="Source samples")
plt.plot(b[:, 0], b[:, 1], "xr", label="Target samples")
plt.legend(loc=0)
plt.title("OT matrix with samples")

# %% ---- 2024-02-23 ------------------------
# Pending
data = []
for i, j in np.argwhere(G0 > 0):
    x1, y1, z1 = a[i, :3]
    x2, y2, z2 = b[j, :3] + 2
    line_group = f"{i}-{j}"
    data.append(dict(x=x1, y=y1, z=z1, line_group=line_group))
    data.append(dict(x=x2, y=y2, z=z2, line_group=line_group))

df = pd.DataFrame(data)
print(df)

# fig = px.line_3d(df, x="x", y="y", z="z", line_group="line_group")

fig = make_subplots(rows=1, cols=1)
trace1 = go.Scatter3d(
    dict(zip(["x", "y", "z"], data1.T[:3])),
    mode="markers",
    name="data1",
    marker=dict(size=1.5, color="red", colorbar=dict()),
)
trace2 = go.Scatter3d(
    dict(zip(["x", "y", "z"], data2.T[:3] + 2)),
    mode="markers",
    name="data2",
    marker=dict(size=1.5, color="green", colorbar=dict()),
)

fig.add_traces([trace1, trace2])

fig.show()

# %%

# %% ---- 2024-02-23 ------------------------
# Pending
