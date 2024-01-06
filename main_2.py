from GMM import GMM
import numpy as np
import os
import matplotlib.pyplot as plt
img = 'Meki.png'
curfolder = os.path.dirname(os.path.abspath(__file__))
im = plt.imread(os.path.join(curfolder, 'images', img))

foreground_slices_list = [(150, 250, 250, 300), (300, 500, 200, 300), (0, 200, 100, 300)]
background_slices_list = [(50, 350, 20, 50), (0, 300, 400, 500), (300, 500, 0, 80)]
like0_im, like1_im = GMM(im, foreground_slices_list, background_slices_list)

fig = plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.title("negative log-likelihoods for Mekkana")
plt.imshow(like1_im)
plt.colorbar()
plt.subplot(222)
plt.title("negative log-likelihoods for background")
plt.imshow(like0_im)
plt.colorbar()
plt.subplot(223)
plt.title("log-likelihood ratio")
plt.imshow(like1_im - like0_im)
plt.colorbar()
# save the plt figure
matplot = 'matplot'
if not os.path.exists(matplot):
    os.makedirs(matplot)
# use os.path.join to join the path and the filename
fig.savefig(os.path.join(curfolder, matplot, 'P1.1-likelihood-ratio.jpg'), dpi=300)

img = 'Meki2.png'
curfolder = os.path.dirname(os.path.abspath(__file__))
im = plt.imread(os.path.join(curfolder, 'images', img))

foreground_slices_list = [(150, 250, 250, 300), (300, 500, 200, 300), (0, 200, 100, 300)]
background_slices_list = [(50, 350, 20, 50), (0, 300, 400, 500), (300, 500, 0, 80)]
like0_im, like1_im = GMM(im, foreground_slices_list, background_slices_list)

fig = plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.title("negative log-likelihoods for Mekkana")
plt.imshow(like1_im)
plt.colorbar()
plt.subplot(222)
plt.title("negative log-likelihoods for background")
plt.imshow(like0_im)
plt.colorbar()
plt.subplot(223)
plt.title("log-likelihood ratio")
plt.imshow(like1_im - like0_im)
plt.colorbar()
# use os.path.join to join the path and the filename
fig.savefig(os.path.join(curfolder, matplot, 'P1.2-likelihood-ratio.jpg'), dpi=300)