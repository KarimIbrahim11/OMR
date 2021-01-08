import numpy as np
import skimage
import skimage.io as io

# Show the figures / plots inside the notebook

from skimage.color import rgb2gray, rgb2hsv, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.util import random_noise
from skimage.measure import find_contours
from skimage.draw import rectangle, rectangle_perimeter
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
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
from statistics import mode

from skimage.io import imsave

import argparse

import sys
from skimage.transform import resize
from skimage.exposure import histogram

from skimage.exposure import histogram
from matplotlib.pyplot import bar
import cv2


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
    plt.show()


def showHist2(histogramImg):
    plt.figure()
    bar(histogramImg[1] * 255, histogramImg[0], width=0.8, align='center')


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


def negativeandSave(path):
    img = read_image(path)
    if path.lower().endswith('.jpg'):
        gray = rgb2gray(img)
    elif path.lower().endswith('.png'):
        gray = rgb2gray(img)
    thresh = threshold_otsu(gray)
    # neg = 1-gray
    normalize = gray > thresh
    # normalize = img_as_ubyte(normalize)
    show_images([img, gray, normalize])
    io.imsave(path + '2.jpg', normalize)


def deskew(gray):
    # image = imread( filename, as_grey=True)
    # threshold to get rid of extraneous noise
    image = gray.copy()
    thresh = threshold_otsu(image)
    print(thresh)
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
    rotation_angle = int(rotation_angle)
    print(rotation_angle)
    '''
    if rotation_angle > 45:
        rotation_number = -(90 - rotation_angle)
    elif rotation_angle < -45:
        rotation_number = 90 - abs(rotation_angle)
    '''
    # show_images([gray,normalize,edges], ["gray", "bin", "edges"])
    rotated = rotate(binary_closing(np.logical_not(normalize), np.ones((3, 3))), rotation_angle, resize=True,
                     mode='constant', cval=0).astype(np.uint8)
    # show_images([gray])
    gray = rotate(gray, rotation_angle, resize=True, mode='constant', cval=255)
    return rotated, gray


def segmentBoxesInImage(boxes, image_to_segment):
    notes_with_lines = []
    for box in boxes:
        [Ymin, Xmin, Ymax, Xmax] = box
        Ymin -= 5
        Ymax += 25
        # print(box)
        rr, cc = rectangle_perimeter(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=image_to_segment.shape)
        # rotated[rr, cc] = 1  # set color white
        notes_with_lines.append(image_to_segment[np.min(rr):np.max(rr), np.min(cc):np.max(cc)])

    notes_with_lines = np.array(notes_with_lines, dtype=object)
    return notes_with_lines


def removeHLines(bin):
    hproj = np.sum(bin, 1)

    # Create output image same height as text, 500 px wide
    m = np.max(hproj)
    w = bin.shape[1]
    result = np.zeros((hproj.shape[0], w))

    # Draw a line for each row
    for row in range(bin.shape[0]):
        cv2.line(result, (0, row), (int(hproj[row] * w / m), row), (255, 255, 255), 1)

    # show_images([bin, result], ['binarized', 'horizontal projection'])

    r, c = result.shape
    for i in range(r):
        if np.sum(result[i]) > 255 * 480:
            bin[i, :] = 0

    # show_images([bin])
    return bin


def get_references(img):
    # Run length encoding
    #     img = binary_dilation(img2,np.ones((3,3)))
    encoded_img = []
    encoded_img_color = []
    # loop on the columns of the img
    for i in range(img.shape[1]):
        col = img[:, i]
        encoded_col = []
        encoded_col_color = []

        current_color = col[0]
        current_count = 0
        # loop on the rows
        for j in range(img.shape[0]):
            if current_color == col[j]:
                current_count += 1
            else:
                # appending count and color
                encoded_col.append(current_count)
                encoded_col_color.append(current_color)

                current_color = col[j]
                current_count = 1
        encoded_col.append(current_count)
        encoded_col_color.append(current_color)

        encoded_img.extend(encoded_col)
        encoded_img_color.extend(encoded_col_color)

    encoded_img = np.array(encoded_img)
    encoded_img_color = np.array(encoded_img_color)

    black_encoded = encoded_img[encoded_img_color == 0]
    white_encoded = encoded_img[encoded_img_color == 1]

    space = mode(black_encoded)
    thickness = mode(white_encoded)

    return space, thickness


def find_stafflines(img, space, thickness):
    row_hist = np.array([sum(img[i, :]) for i in range(img.shape[0])])
    #     thickness += 2
    print(max(row_hist))

    staff_indices = []

    staff_length = 5 * (space + thickness) - space
    row = 0
    thresh = 0.6
    print(img.shape)

    staff_lines = row_hist > thresh * img.shape[1]
    staff_indices = np.where(staff_lines == True)[0]

    #     while row< (img.shape[0] - staff_length + 1):
    #         staff_lines = [row_hist[j:j+thickness] for j in range(row,
    #                             row + (4)*(thickness + space)+1, thickness+space)]

    #         for staff in staff_lines:
    #             if sum(staff)/thickness < thresh*img.shape[1]:
    #                 row += 1
    #                 break
    #             else:
    #                 print(row,sum(staff),thresh*img.shape[1])
    #                 staff_row_indices = [list(range(j, j + thickness)) for j in
    #                                      range(row,
    #                                        row + (4) * (thickness + space) + 1,
    #                                        thickness + space)]
    #                 staff_indices.append(staff_row_indices)
    #                 row += staff_length
    #                 break

    return staff_indices


def find_verticalLines(img):
    col_hist = np.array([sum(img[:, i]) for i in range(img.shape[1])])
    thresh = 0.6
    staff_lines = col_hist > thresh * img.shape[0]
    staff_indices = np.where(staff_lines == True)[0]
    return staff_indices


def find_horizontalLines(img):
    row_hist = np.array([sum(img[i, :]) for i in range(img.shape[0])])
    print(max(row_hist))
    thresh = 0.6
    print(img.shape)
    staff_lines = row_hist > thresh * img.shape[1]
    staff_indices = np.where(staff_lines == True)[0]
    return staff_indices


def countStems(stems_indices):
    stems = []
    for i in range(len(stems_indices)):
        temp_stem = stems_indices[i]
        if i == 0 or temp_stem > stems_indices[i - 1] + 1:
            stems.append(temp_stem)
    return stems, len(stems)


def draw_contours(img):
    # se = np.ones((3, 3))
    # closing_card = binary_erosion(binary_dilation(new_image, se), se)
    contours = find_contours(img, 0.8)
    # print("contours:", len(contours))
    bounding_boxes = np.array([[min(c[:, 1]), max(c[:, 1]), min(c[:, 0]), max(c[:, 0])] for c in contours]).astype(int)
    # print("bounding_boxes:", bounding_boxes)
    img_boxes = img.copy().astype(float)
    gray = rgb2gray(img)

    # When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside
    # boxes in img_with_boxes
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        # if (Xmax - Xmin) / (Ymax - Ymin) < 2.5 or (Xmax - Xmin) / (Ymax - Ymin) > 3.5:
        #     print("continues")
        #     continue
        # print("box:", box)
        rr, cc = rectangle_perimeter(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=gray.shape)
        img_boxes[rr, cc] = 1  # set color white

    return img_boxes, bounding_boxes


# CCA APPROACH TESTED
def CCA(binary):
    # Perform CCA on the mask
    labeled_image = skimage.measure.label(binary, connectivity=2, return_num=True, background=0)
    components = skimage.measure.regionprops(labeled_image[0])

    return components


# CCA Display Components TESTED
def displayComponents(binary, components):
    # takes ski.image.regionProps output
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(binary)
    for component in components:
        # take regions with large enough areas
        if component.area >= 30:
            # draw rectangle around segmented coins
            # print("orientation of component:",component.orientation)
            minR, minC, maxR, maxC = component.bbox
            rect = mpatch.Rectangle((minC, minR), maxC - minC, maxR - minR, fill=False, edgecolor='blue', linewidth=2)
            # show_images([component.image], ["el sorraaa"])
            ax.add_patch(rect)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# Retrieve Boxes bs
def RetrieveComponentBox(components):
    boxes = []
    for component in components:
        # take regions with large enough areas
        if component.area >= 30:
            # draw rectangle around segmented coins
            boxes.append(component.bbox)
    return np.array(boxes, dtype=object)


# Convert Each Component to image and append them in a single array
def componentsToImages(components):
    images = []
    for component in components:
        # take regions with large enough areas
        if component.area >= 30:
            # draw rectangle around segmented coins
            minR, minC, maxR, maxC = component.bbox
            rect = mpatch.Rectangle((minC, minR), maxC - minC, maxR - minR, fill=False, edgecolor='red', linewidth=2)
            images.append(component.image)
    return np.array(images, dtype=object)
