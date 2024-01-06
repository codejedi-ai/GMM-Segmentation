# loading standard modules
import numpy as np
import matplotlib.pyplot as plt
import maxflow
from skimage import img_as_ubyte
from sklearn import mixture
from skimage.color import rgb2gray
import os
'''
def wpq(lambda_, p, q, sigma):
  return lambda_ * np.exp(- ((p - q) ** 2) / sigma ** 2)

def wpq_vector(lambda_, p, q, sigma):
    return lambda_ * np.exp(-(np.linalg.norm(p - q, axis = 2) ** 2) / sigma ** 2)

def wpq_vector2(lambda_, p, q, sigma):
    d = p-q
    return lambda_ * np.exp(-((d[...,0]**2)+(d[...,1]**2)+(d[...,2]**2)) / sigma ** 2)
'''
class MyGraphCuts:
    bgr_value = 0
    obj_value = 1
    none_value = 2

    def __init__(self, img, sigma, lambda_):

        self.num_rows = img.shape[0]
        self.num_cols = img.shape[1]

        self.sigma = sigma
        self.img = img_as_ubyte(img)
        self.lambda_ = lambda_
        self.gmm1 = mixture.GaussianMixture(n_components=7, covariance_type='full')
        self.gmm0 = mixture.GaussianMixture(n_components=7, covariance_type='full')
        self.g, self.nodeids = self.calculate_graph()

    import numpy as np

    def _extract_colors(self, foreground_slices_list, background_slices_list):
        im = self.img
        # Extract the RGB channels from the image
        R = im[:, :, 0].flatten()
        G = im[:, :, 1].flatten()
        B = im[:, :, 2].flatten()

        # Initialize empty lists for foreground and background colors
        R_foreground = []
        G_foreground = []
        B_foreground = []
        R_background = []
        G_background = []
        B_background = []

        # Extract the foreground colors using the given slices
        for foreground_slices in foreground_slices_list:
            R_foreground.extend(
                im[foreground_slices[0]:foreground_slices[1], foreground_slices[2]:foreground_slices[3], 0].flatten())
            G_foreground.extend(
                im[foreground_slices[0]:foreground_slices[1], foreground_slices[2]:foreground_slices[3], 1].flatten())
            B_foreground.extend(
                im[foreground_slices[0]:foreground_slices[1], foreground_slices[2]:foreground_slices[3], 2].flatten())

        # Extract the background colors using the given slices
        for background_slices in background_slices_list:
            R_background.extend(
                im[background_slices[0]:background_slices[1], background_slices[2]:background_slices[3], 0].flatten())
            G_background.extend(
                im[background_slices[0]:background_slices[1], background_slices[2]:background_slices[3], 1].flatten())
            B_background.extend(
                im[background_slices[0]:background_slices[1], background_slices[2]:background_slices[3], 2].flatten())

        # Return the extracted colors
        # return (R_foreground, G_foreground, B_foreground), (R_background, G_background, B_background)

        # foreground_colors, background_colors = extract_colors(im, foreground_slices_list, background_slices_list)

        R1, G1, B1 = (R_foreground, G_foreground, B_foreground)
        R0, G0, B0 = (R_background, G_background, B_background)
        # set of colors for all image pixels
        R = im[:, :, 0].flatten()
        G = im[:, :, 1].flatten()
        B = im[:, :, 2].flatten()
        data1 = np.vstack([R1, G1, B1])
        data0 = np.vstack([R0, G0, B0])
        data = np.vstack([R, G, B])
        return data0, data1, data

    def _GMM(self, data0, data1, data):
        im = self.img
        # estimate color distributions using GMM - should be fast (1-2 seconds for both models)
        from sklearn import mixture

        gmm1 = self.gmm1
        gmm0 = self.gmm0
        gmm1.fit(data1.T)
        gmm0.fit(data0.T)
        like1 = -np.array(gmm1.score_samples(data.T))
        like0 = -np.array(gmm0.score_samples(data.T))
        like1_im = like1.reshape((im.shape[0], im.shape[1]))
        like0_im = like0.reshape((im.shape[0], im.shape[1]))
        # always forground then background 0 then 1
        return like0_im, like1_im
    def calculate_graph(self):
        img = self.img
        lambda_ = self.lambda_
        sigma = self.sigma
        # img = np.sum(img, axis = 2) / 3
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes(img.shape[:2])

        def wpq_vector(lambda_, p, q, sigma):
            return lambda_ * np.exp(-(np.linalg.norm(p - q, axis=2) ** 2) / sigma ** 2)

        # roll the image to the right and down
        img_roll_right = np.roll(img, -1, axis=1)
        img_roll_down = np.roll(img, -1, axis=0)
        # calculate the weights
        # weights_right = wpq_vector(lambda_, img, img_roll_right, sigma)
        # weights_down = wpq_vector(lambda_, img, img_roll_down, sigma)
        n_right = wpq_vector(lambda_, img, img_roll_right, sigma)
        n_below = wpq_vector(lambda_, img, img_roll_down, sigma)

        structure_x, structure_y = np.zeros((3, 3)), np.zeros((3, 3))
        structure_x[1, 2] = 1
        structure_x[1, 0] = 1
        structure_y[2, 1] = 1
        structure_y[0, 1] = 1

        # g.add_grid_edges(nodeids, weights=n_right, structure=structure_x, symmetric=True)
        # g.add_grid_edges(nodeids, weights=n_below, structure=structure_y, symmetric=True)
        # symmetric = False
        g.add_grid_edges(nodeids, n_right, structure=structure_x, symmetric=False)
        g.add_grid_edges(nodeids, n_below, structure=structure_y, symmetric=False)
        return g, nodeids

    def compute_labels_mask(self, seed_mask):
        # seed_mask = np.sum(seed_mask, axis = 2) / 3
        # +---------+---------+
        # |         |         |
        # |   bgr   |  none   |
        # |         |         |
        # +---------+---------+
        # |         |         |
        # |  none   |   obj   |
        # |         |         |
        # +---------+---------+
        foreground_slices_list, background_slices_list = seed_mask
        data0, data1, data = self._extract_colors(foreground_slices_list, background_slices_list)
        like0_im, like1_im = self._GMM(data0, data1, data)
        g, nodeids = self.g, self.nodeids
        # print(n_right)
        # print(n_below)
        g.add_grid_tedges(nodeids, like0_im, like1_im)
        flow = g.maxflow()
        print(f"Maximum flow: {flow}")
        sgm = g.get_grid_segments(nodeids)
        return sgm  # segments
    def compute_labels(self, seed_mask):
        # seed_mask = np.sum(seed_mask, axis = 2) / 3
        # +---------+---------+
        # |         |         |
        # |   bgr   |  none   |
        # |         |         |
        # +---------+---------+
        # |         |         |
        # |  none   |   obj   |
        # |         |         |
        # +---------+---------+
        data0, data1 = seed_mask
        im = self.img
        g, nodeids = self.g, self.nodeids
        R = im[:, :, 0].flatten()
        G = im[:, :, 1].flatten()
        B = im[:, :, 2].flatten()
        data = np.vstack([R, G, B])
        like0_im, like1_im = self._GMM(data0, data1, data)
        # print(n_right)
        # print(n_below)
        g.add_grid_tedges(nodeids, like0_im, like1_im)
        flow = g.maxflow()
        print(f"Maximum flow: {flow}")
        sgm = g.get_grid_segments(nodeids)
        return sgm









