from MyGraphCuts import MyGraphCuts
import matplotlib.pyplot as plt
import os
img = 'Meki2.png'
curfolder = os.path.dirname(os.path.abspath(__file__))
im_folder = os.path.join(curfolder, 'images')
# iterate through all the images in the folder
# get the list of images
img_list = os.listdir(im_folder)
print(img_list)
for i in range(len(img_list)):
    im = plt.imread(os.path.join(im_folder, img_list[i]))
    # display the graph
    # set the center of the image to be the source
    # set the edges to be the weights

    segmenter = MyGraphCuts(im, 0.01, 0.005)

    foreground_slices_list = [(150, 250, 250, 300), (300, 500, 200, 300), (0, 200, 100, 300)]
    background_slices_list = [(50, 350, 20, 50), (0, 300, 400, 500), (300, 500, 0, 80)]

    labels = segmenter.compute_labels_mask((foreground_slices_list, background_slices_list))
    # get the name of the img
    img_name = img_list[i].split('.')[0]
    plt.imsave(os.path.join(curfolder, "matplot", f'{img_name}_labels.png'), labels)