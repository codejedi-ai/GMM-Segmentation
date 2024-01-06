# loading standard modules
import numpy as np
import matplotlib.pyplot as plt
import maxflow
from skimage import img_as_ubyte
from sklearn import mixture
from skimage.color import rgb2gray
import os
import GMM
def wpq(lambda_, p, q, sigma):
  return lambda_ * np.exp(- ((p - q) ** 2) / sigma ** 2)

def wpq_vector(lambda_, p, q, sigma):
    return lambda_ * np.exp(-(np.linalg.norm(p - q, axis = 2) ** 2) / sigma ** 2)

def wpq_vector2(lambda_, p, q, sigma):
    d = p-q
    return lambda_ * np.exp(-((d[...,0]**2)+(d[...,1]**2)+(d[...,2]**2)) / sigma ** 2)

img1 = plt.imread('images/Meki.png')
wpq_vector(1, img1, np.roll(img1, -1, axis = 1), 5).shape


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

        self.g, self.nodeids = self.calculate_graph()

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
        n_right = wpq_vector(lambda_, img, np.roll(img, -1, axis=1), sigma)
        n_below = wpq_vector(lambda_, img, np.roll(img, -1, axis=0), sigma)

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
        data0, data1, data = GMM.extract_colors(self.img, foreground_slices_list, background_slices_list)
        like0_im, like1_im = GMM.GMM(self.img, data0, data1, data)
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

        R = im[:, :, 0].flatten()
        G = im[:, :, 1].flatten()
        B = im[:, :, 2].flatten()
        data = np.vstack([R, G, B])
        like0_im, like1_im = GMM.GMM(self.img, data0, data1, data)
        # print(n_right)
        # print(n_below)
        g.add_grid_tedges(nodeids, like0_im, like1_im)
        flow = g.maxflow()
        print(f"Maximum flow: {flow}")
        sgm = g.get_grid_segments(nodeids)
        return sgm






img = 'Meki2.png'
curfolder = os.path.dirname(os.path.abspath(__file__))
im = plt.imread(os.path.join(curfolder, 'images', img))
# display the graph
# set the center of the image to be the source
# set the edges to be the weights


graphcut = MyGraphCuts(im, 0.01, 0.005)

foreground_slices_list = [(150, 250, 250, 300), (300, 500, 200, 300), (0, 200, 100, 300)]
background_slices_list = [(50, 350, 20, 50), (0, 300, 400, 500), (300, 500, 0, 80)]

g, nodeids = graphcut.g, graphcut.nodeids

labels = graphcut.compute_labels_mask((foreground_slices_list, background_slices_list))
plt.imsave(os.path.join(curfolder, "matplot", 'labels.png'), labels)





'''z

img1 = plt.imread('images/Meki.png')
app = MyGraphCuts(img1, 50, 10)
app.run()



img1 = plt.imread('images/bunny.bmp')
app = MyGraphCutsCoulorBasedLikelihood(img1, 50, 20, [1,2,3], [1,2,3])
app.run()


sigma = [0.01, 0.1, 1, 10, 100]
img1 = plt.imread('images/bunny.bmp')
print('Sigma used:', sigma[1])
app = MyGraphCuts(img1, sigma[0], 70, 40)
app.run()

img2 = plt.imread('images/lama.jpg')
app = MyGraphCuts(img2, sigma = 1, lambda_ = 0)
app.run()

img2 = plt.imread('images/lama.jpg')
app = MyGraphCuts(img2, sigma = 1, 3)
app.run()

'''