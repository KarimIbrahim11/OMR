import numpy as np
import skimage.io as io

# Show the figures / plots inside the notebook

from skimage.color import rgb2gray, rgb2hsv, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.util import random_noise
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math as m
import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from scipy import fftpack
from scipy.signal import convolve2d
from skimage.util import random_noise
from skimage.filters import median, gaussian, threshold_otsu
from skimage.filters import roberts, sobel, sobel_h, scharr
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, skeletonize, thin
from skimage import img_as_ubyte
from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import os

from skimage.io import imsave

import argparse

import sys
from skimage.transform import resize
from skimage.exposure import histogram

from skimage.exposure import histogram
from matplotlib.pyplot import bar


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
    plt.show()


def showHist2(histogramImg):
    plt.figure()
    bar(histogramImg[1]*255, histogramImg[0], width=0.8, align='center')

def read_image(path):
    return io.imread(path)


def otsu_binarize(gray_img):
    thresh = threshold_otsu(gray_img)
    binary = gray_img > thresh
    return binary.astype(int), thresh


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


'''
def drawLine(ax, angle, dist):
   
    This function should draw the lines, given axis(ax), the angle and the distance parameters

    TODO:
    Get x1,y1,x2,y2
 
    angle_deg = 180 * abs(angle) / np.pi

    # if abs(angle) >= ((40 / 180) * np.pi) and abs(angle) <= ((60 / 180) * np.pi):
    #     print("huraay")
    #
    x1, y1 = 0, (dist / m.cos(angle))
    x2, y2 = (dist / m.sin(angle)), 0

    # This line draws the line in red

    ax[1].plot((x1, y1), (x2, y2), '-r')


# image = rgb2gray(io.imread('triangles.png'))


def image_lines(image, thres=None):
    edges = canny(image)

    show_images([image, edges],['image','edges'])
    hough_space, angles, distances = hough_line(edges)
    if thres == None:
        thres = 0.5 * np.max(hough_space)
    print(thres)

    acumm, angles, dists = hough_line_peaks(hough_space, angles, distances, threshold=thres)

    ## This part draw the lines on the image.

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap=cm.gray)
    for angle, dist in zip(angles, dists):
        drawLine(ax, angle, dist)
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

'''


def deskew(image):
    #image = imread(filename, as_grey=True)

    # threshold to get rid of extraneous noise
    thresh = threshold_otsu(image)
    normalize = image > thresh

    # gaussian blur
    blur = gaussian(normalize, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 9999 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_angle = histo[1][np.argmax(histo[0])]
    print(rotation_angle)
    '''
    if rotation_angle > 45:
        rotation_number = -(90 - rotation_angle)
    elif rotation_angle < -45:
        rotation_number = 90 - abs(rotation_angle)
    '''
    show_images([gray,normalize,edges], ["gray", "bin", "edges"])
    rotated = rotate(binary_closing(np.logical_not(normalize), np.ones((3, 3))), rotation_angle, resize=True, mode='constant', cval=0)
    # counts, bins = np.histogram(rotated)
    # plt.hist(bins[:-1], bins, weights=counts, orientation='horizontal')
    return rotated



# if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
path = '27_!.jpg'
img = read_image(path)
gray = rgb2gray(img)

rotated = deskew(gray)

# gray = median(gray)
# binImg = otsu_binarize(rotated)
# edges = canny(binImg)
# show_images([gray, edges],['image','edges'])
# binImg = binary_closing(binImg)
# image_lines(gray, thres=90)

photo = resize(rotated, (256, 256))
hist,_ = histogram(photo, nbins=256)
# showHist(hist)

