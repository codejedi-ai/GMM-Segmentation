# README

GMM segmentation is a Python package that uses PyMaxflow and Gaussian mixture modelling to perform image segmentation. Image segmentation is the process of dividing an image into meaningful regions or segments, such as foreground and background, objects and scenes, or different tissue types in medical images. GMM segmentation uses a probabilistic approach that assumes each pixel in the image belongs to one of a finite number of Gaussian distributions, each representing a segment. PyMaxflow is a library that implements the maxflow/mincut algorithm, which is used to find the optimal partition of the image into segments based on the Gaussian mixture model. GMM segmentation can handle various types of images, such as grayscale, color, or multispectral, and can also incorporate spatial information and regularization terms to improve the segmentation results.

To use GMM segmentation, you need to install PyMaxflow and its dependencies, such as NumPy and Cython. You can install PyMaxflow using pip:

```bash
pip install PyMaxflow
```

You also need to install GMM segmentation from its GitHub repository:

```bash
git clone https://github.com/user/GMM-segmentation.git
cd GMM-segmentation
pip install -f requirements.txt
```

To segment an image, you need to import the GMM_segmentation module and create an instance of the GMM_segmenter class. You can specify the number of segments, the type of covariance matrix, the initialization method, and other parameters in the constructor. For example, to create a segmenter with 3 segments, full covariance matrix, and k-means initialization, you can use:

```python
from GMM_segmentation import GMM_segmenter
segmenter = GMM_segmenter(n_components=3, covariance_type='full', init_params='kmeans')
```

Then, you can use the fit method to fit the Gaussian mixture model to the image data, and the predict method to assign each pixel to a segment. For example, to segment an image stored in a NumPy array called image, you can use:

```python
segmenter.fit(image)
labels = segmenter.predict(image)
```

The labels array will contain the segment index (from 0 to n_components-1) for each pixel in the image. You can use the labels array to visualize the segmentation result, or to perform further analysis on the segments. You can also use the score method to evaluate the log-likelihood of the image data given the fitted model, or the bic method to compute the Bayesian information criterion, which can be used to compare different models or select the optimal number of segments.

For more details and examples, please refer to the documentation and the demo notebook in the GitHub repository. If you have any questions or feedback, please feel free to contact the author at user@email.com. Thank you for using GMM segmentation! 😊


The difference between the pixel intensites are put through the function:
$$
\text{wpq_vector}(\lambda_, p, q, \sigma)= \lambda \times \exp\left(-\frac{\|p - q\|_2^2}{\sigma^2}\right)
$$

Which forms a meshed grid between every neighboring horizontal pixels and every neighboring vertical pixels. 

The function `compute_labels_mask` requires regions of the image to segment, specified as a tuple in the form `(x1, x2, y1, y2)`. These coordinates define the rectangular region of interest within the image. You can provide these values to the function to perform the desired segmentation task. If you have any further questions or need additional assistance, feel free to ask! 😊

# Read me is still WIP
