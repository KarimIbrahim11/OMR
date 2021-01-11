import numpy as np
import skimage
import skimage.io as io

# Show the figures / plots inside the notebook

from skimage.color import rgb2gray, rgb2hsv, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny, match_template
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_local, threshold_niblack
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
    # OTSU
    # thresh = threshold_otsu(image)
    # SAUVOLA
    thresh = threshold_sauvola(image, window_size=51)
    # NIBLACK
    # thresh = threshold_niblack(image, window_size=25, k=0.8)
    # LOCAL
    # bin1 =  threshold_local(image, 3, 'mean')
    # func = lambda arr: arr.mean()
    # thresh = threshold_local(image, 11, 'generic', param=func)

    # print(thresh)
    normalize = image > thresh
    show_images([normalize], ["binary"])
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
    thresh = 0.7
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
    thisdict = {}
    sorted_notes_images = []
    index = 0
    keys = []
    for component in components:
        if component.area >= 44:
            minR, minC, maxR, maxC = component.bbox
            thisdict[minC] = []
            thisdict[minC].append(binary[minR:maxR+2, minC:maxC+2])
            keys.append(str(index))
            index += 1
    print(thisdict.keys())
    for key in sorted(thisdict.keys()):
        sorted_notes_images.append(thisdict[key][0])
    return components, sorted_notes_images

def componentsAreas(components):
    area = []
    for component in components:
        area.append(component.area)
    return np.array(area)
# CCA Display Components TESTED
def displayComponents(binary, components):
    # takes ski.image.regionProps output
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(binary)
    for component in components:
        # take regions with large enough areas
        if component.area >= 44:
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
def RetrieveComponentBox(images):
    boxes = []
    for img in images:
        minR = np.min(img[0])
        maxR = np.max(img[0])
        minC = np.min(img[1])
        maxC = np.max(img[1])
        bbox = [minR, minC, maxR, maxC]
        boxes.append(bbox)
    return np.array(boxes, dtype=object)


# Convert Each Component to image and append them in a single array
def componentsToImages(components):
    images = []
    for component in components:
        # take regions with large enough areas
        if component.area >= 44:
            # draw rectangle around segmented coins
            minR, minC, maxR, maxC = component.bbox
            rect = mpatch.Rectangle((minC, minR), maxC - minC, maxR - minR, fill=False, edgecolor='red', linewidth=2)
            images.append(component.image)
    return np.array(images, dtype=object)


# SKIMAGE TEMPLATE MATCHING
def template_Match(img, template):
    result = match_template(img, template)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(template, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('image')

    ax2.imshow(img, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('rotated')
    # highlight matched region
    hcoin, wcoin = template.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()


def findBoundingCircleArea(img, contours):
    (x, y), radius = cv2.minEnclosingCircle(contours[0])
    center = (int(x), int(y))
    area = m.pi * int(radius) * int(radius)
    bounding_circle = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), center, int(radius), (0, 255, 0), 2)
    return area, bounding_circle


def findContourArea(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        area = cv2.contourArea(contours[0])
    else:
        # print("No contours found")
        area = 0
    return area, contours


def extract_features_for_head_tail(img):
    area, contours = findContourArea(img)
    if area > 10 and len(contours) != 0:
        area1, _ = findBoundingCircleArea(img, contours)
        ratioCirc = area / area1
    else:
        ratioCirc = 0
    feature = ratioCirc
    return feature


def bin_image_to_opencvimage(bin_img):
    # print(bin_img.shape)
    shape = np.asarray(bin_img.shape)
    shape = np.append(shape, 3)
    # print(shape)
    wawa = np.zeros(shape)
    # print(np.max(bin_img))
    wawa[:, :, 0] = np.array(bin_img * 255, dtype=np.uint8)
    wawa[:, :, 1] = np.array(bin_img * 255, dtype=np.uint8)
    wawa[:, :, 2] = np.array(bin_img * 255, dtype=np.uint8)
    # print("wawa:", wawa.shape)

    # wawa = np.negative(wawa)
    wawa = cv2.normalize(wawa, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    gray = cv2.cvtColor(wawa, cv2.COLOR_BGR2GRAY)
    kernelSize = (3, 3)
    blur = cv2.blur(gray, kernelSize)
    _, binary = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    return binary


def classifyNotePositionInSegment(img):
    top_image = img[0:img.shape[0] // 2, :]
    bot_image = img[img.shape[0] // 2: img.shape[0] - 1, :]
    # show_images([top_image, bot_image],["Top image", "Bot Image"])
    top_image_cv = bin_image_to_opencvimage(top_image)
    bot_image_cv = bin_image_to_opencvimage(bot_image)

    ratio_top = extract_features_for_head_tail(top_image_cv)
    ratio_bot = extract_features_for_head_tail(bot_image_cv)
    print("Circle Ratios: ", [ratio_top, ratio_bot])
    tb = -1
    if max(ratio_bot, ratio_top) == ratio_top:
        print("Note head is up")
        tb = 0
    else:
        print("Note head is down")
        tb = 1
    return tb, top_image, bot_image


###### warping functions
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

##################### USING WARP PERSPECTIVE
'''
thresh = threshold_sauvola(image, window_size=51)
binary = image > thresh
#blur = gaussian(binary, 3)
#edges = canny(blur)
#print(edges.shape)
#edged = bin_image_to_opencvimage(edges)
# edged = np.resize(edged, (-1, 300))
np.resize(binary, (binary.shape[0]//3, binary.shape[1]//3))
edged = bin_image_to_opencvimage(binary)
ratio = edged.shape[0] / 300.0
orig = edged.copy()
# edged = imutils.resize(edged, height = 300)
# convert the image to grayscale, blur it, and find edges
# in the image

gray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
# show_images([edged], ["edged"])
print(edged.shape)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        print("Ana hena ya salama")
        screenCnt = approx
        break
cv2.drawContours(edged, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Game Boy Screen", image)
cv2.waitKey(0)
# gaussian blur
# displayComponents(binary, componentsss)
# canny edges in scikit-image
# edges = canny(blur)

# negativeandSave('quarternote.png')
'''