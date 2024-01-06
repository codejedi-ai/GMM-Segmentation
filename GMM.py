import numpy as np


def extract_colors(im, foreground_slices_list, background_slices_list):
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


def GMM(im, data0, data1, data):

    # estimate color distributions using GMM - should be fast (1-2 seconds for both models)
    from sklearn import mixture

    gmm1 = mixture.GaussianMixture(n_components=7, covariance_type='full')
    gmm1.fit(data1.T)
    gmm0 = mixture.GaussianMixture(n_components=7, covariance_type='full')
    gmm0.fit(data0.T)
    like1 = -np.array(gmm1.score_samples(data.T))
    like0 = -np.array(gmm0.score_samples(data.T))
    like1_im = like1.reshape((im.shape[0], im.shape[1]))
    like0_im = like0.reshape((im.shape[0], im.shape[1]))
    # always forground then background 0 then 1
    return like0_im, like1_im