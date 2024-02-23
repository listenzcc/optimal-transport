"""
File: image-example.py
Author: Chuncheng Zhang
Date: 2024-02-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Image example

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-23 ------------------------
# Requirements and constants
import io
import skdim
import requests
import numpy as np

import ot
import ot.plot
import matplotlib.pyplot as plt

from PIL import Image
from collections import UserDict
from IPython.display import display


# %% ---- 2024-02-23 ------------------------
# Function and class


class AttrDict(UserDict):
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        if key == "data":
            return super().__setattr__(key, value)
        return self.__setitem__(key, value)


def get_image(category: str = "sport"):
    # width x height
    size = (400, 300)

    sz = "x".join([str(e) for e in size])
    resp = requests.get(f"https://source.unsplash.com/random/{sz}?{category}")
    img = Image.open(io.BytesIO(resp.content))
    mat = np.array(img) / 255.0

    mat = mat.reshape((size[0] * size[1], 3))

    def mat2im(mat):
        return np.reshape(np.clip(mat, 0, 1) * 255, (size[1], size[0], 3)).astype(
            np.uint8
        )

    return AttrDict(img=img, mat=mat, mat2im=mat2im)


# %% ---- 2024-02-23 ------------------------
# Play ground
ad1 = get_image(category="flower")
ad2 = get_image(category="sport")
display(ad1.img)
display(ad2.img)


# %% ---- 2024-02-23 ------------------------
# Pending
# Training samples
n = 500

I1 = np.array(ad1.img)
I2 = np.array(ad2.img)

X1 = ad1.mat
X2 = ad2.mat

idx1 = np.random.randint(X1.shape[0], size=(n,))
idx2 = np.random.randint(X1.shape[0], size=(n,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]

# %%
# Training and transport
# EMDTransport
ot_emd = ot.da.EMDTransport()
ot_emd.fit(Xs=Xs, Xt=Xt)

# SinkhornTransport
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

# prediction between images (using out of sample prediction as in [6])
transp_Xs_emd = ot_emd.transform(Xs=X1)
transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)

transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
transp_Xt_sinkhorn = ot_sinkhorn.inverse_transform(Xt=X2)

I1t = ad1.mat2im(transp_Xs_emd)
I2t = ad2.mat2im(transp_Xt_emd)

I1te = ad1.mat2im(transp_Xs_sinkhorn)
I2te = ad2.mat2im(transp_Xt_sinkhorn)


# %%
# Images
plt.figure(1, figsize=(6.4, 3))

plt.subplot(1, 2, 1)
plt.imshow(I1)
plt.axis("off")
plt.title("Image 1")

plt.subplot(1, 2, 2)
plt.imshow(I2)
plt.axis("off")
plt.title("Image 2")

plt.tight_layout()
plt.show()

# %%
# Samples
plt.figure(2, figsize=(6.4, 3))

plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 2], c=Xs)
plt.axis([0, 1, 0, 1])
plt.xlabel("Red")
plt.ylabel("Blue")
plt.title("Image 1")

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 2], c=Xt)
plt.axis([0, 1, 0, 1])
plt.xlabel("Red")
plt.ylabel("Blue")
plt.title("Image 2")

plt.tight_layout()
plt.show()

# %%
# Result
plt.figure(3, figsize=(6, 8))

data = [
    (1, I1, "Image 1"),
    (2, I2, "Image 2"),
    (3, I1t, "Image 1 Adapt"),
    (4, I2t, "Image 2 Adapt"),
    (5, I1te, "Image 1 Adapt (reg)"),
    (6, I2te, "Image 2 Adapt (reg)"),
]

for j, im, title in data:
    plt.subplot(3, 2, j)
    plt.imshow(im)
    plt.axis("off")
    plt.title(title)

plt.tight_layout()
plt.show()


# %% ---- 2024-02-23 ------------------------
# Pending

# %%
