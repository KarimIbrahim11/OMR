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
# Always make all imports in the first cell of the notebook, run them all once.
import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import skimage.io as io
from mpl_toolkits.mplot3d import Axes3D

from skimage.color import rgb2gray, rgb2hsv
from scipy import fftpack
from scipy.signal import convolve2d
import skimage
from skimage.util import random_noise
from skimage.filters import median, gaussian, threshold_otsu
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.filters import roberts, sobel, sobel_h, scharr
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, skeletonize, thin
from skimage import img_as_ubyte
from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import moments_hu, moments_central, moments_normalized, moments

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
    thresh = threshold_sauvola(image, window_size=61)
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


def countStaffLines(staff_indices):
    staffs = []
    for i in range(len(staff_indices)):
        temp_staff = staff_indices[i]
        if i == 0 or temp_staff > staff_indices[i - 1] + 1:
            staffs.append(temp_staff)
    return staffs, len(staffs)

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
    boxes = []
    areas_over_bbox = []
    for component in components:
        if component.area >= 44:
            minR, minC, maxR, maxC = component.bbox
            thisdict[minC] = []
            thisdict[minC].append(binary[minR:maxR + 2, minC:maxC + 2])
            thisdict[minC].append(component.bbox)
            thisdict[minC].append(component.area / component.bbox_area)
            keys.append(str(index))
            index += 1
    print(thisdict.keys())
    for key in sorted(thisdict.keys()):
        sorted_notes_images.append(thisdict[key][0])
        boxes.append(thisdict[key][1])
        areas_over_bbox.append(thisdict[key][2])
    return components, sorted_notes_images, boxes, areas_over_bbox


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


'''
# Retrieve Boxes bs
def RetrieveComponentBox(images):
    boxes = []
    for img in images:
        minR = np.amin(img, axis=0)
        maxR = np.amax(img, axis=0)
        minC = np.amin(img, axis=1)
        maxC = np.amax(img, axis=1)
        bbox = [minR, minC, maxR, maxC]
        boxes.append(bbox)
    return np.array(boxes, dtype=object)
'''

# Convert Each Component to image and append them in a single array
'''
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

'''


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


def split_images(img, s):
    imgs = []
    row_hist = np.array([sum(img[i, :]) for i in range(img.shape[0])])

    for i in range(len(row_hist) - 4 * s):
        if row_hist[i] <= 0.03 * img.shape[1] < row_hist[i + 1]:
            start = i
            break
    if start < s:
        start = s

    cum_hist = np.array([sum(row_hist[i:i + s]) for i in range(start, len(row_hist), s)])

    # print(cum_hist)

    thresh = img.shape[1] * s * 0.01
    # print(thresh)

    temp = start
    last = temp + 7 * s
    for c in range(0, len(cum_hist)):
        if c == len(cum_hist) - 1:
            # print("pppp")
            # print("temp", temp, "last", last)
            # print(last - temp)
            if last - temp > 4 * s:
                imgs.append(img[temp - s:last + s])
        elif cum_hist[c] <= thresh < cum_hist[c + 1]:
            # print("hhhh")
            # print((c * s) + start - temp)
            if int(c * s) + start - temp > 4 * s:
                imgs.append(img[temp - s:int(c * s) + start + s])
            temp = int(c * s) + start
            last = 0
        elif cum_hist[c] <= thresh and last == 0:
            last = c * s + start

    return imgs


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


##################### KNN

def training_data(shapes):
    x_train = []
    y_train = []
    for i in range(len(shapes)):
        str1 = 'KNN Attempt/images/'
        str2 = i
        str3 = '/*'
        str4 = str1 + str(str2) + str3
        for filename in sorted(glob.glob(str4)):
            print(filename)
            img = read_training_image(filename)  ## cv2.imread reads images in RGB format
            x_train.append(img)
            y_train.append(i)
    return x_train, y_train


def find_regionprop(img):
    labeled_image = skimage.measure.label(img, connectivity=2, return_num=True, background=1)
    components = skimage.measure.regionprops(labeled_image[0])
    return components


def regionVolume(region):
    area = region.shape[0] * region.shape[1]
    b_area = len(region == 0)
    # print("Black area= ", b_area)
    if area == 0:
        area = 1
    vol = b_area / area
    return vol


def region16(image):
    regions = []
    for i in range(2):
        for j in range(2):
            regions.append(image[i * image.shape[0] // 2: (i + 1) * image.shape[0] // 2,
                           j * image.shape[1] // 2: (j + 1) * image.shape[1] // 2])

    return regions


def extract_features(components):
    features = []
    # print("ana hena ya salama",len(components))
    for component in components:
        if component.area >= 44:
            # show_images([component.image])
            # feature 1
            features.append(component.area / component.bbox_area)
            features.append(component.image.shape[0] / component.image.shape[1])
            # feature 2
            moments = cv2.HuMoments(cv2.moments(component.image.astype(np.uint8))).flatten()
            # print(m)
            # mu = moments_central(component.image)
            # nu = moments_normalized(mu)
            # moments = moments_hu(nu)

            # for moment in moments:
            # print(len(moments_hu(nu)))
            # print(len(moments))
            for moment in moments:
                features.append(moment)
            # feature 3
            regions = region16(component.image)
            for region in regions:
                volume = regionVolume(region)
                features.append(volume)
            # print("features:",len(features))
    return features


def extract_features_single_img(img, area_over_bbox):
    features = []
    # print("ana hena ya salama",len(components))
    area = img.shape[0] * img.shape[1]
    if area >= 44:
        # show_images([component.image])
        '''
        minR, minC, maxR, maxC = box
        height = maxR - minR
        width = maxC - minC
        bbox_area = height*width
        '''
        # feature 1
        features.append(area_over_bbox)
        features.append(img.shape[0] / img.shape[1])
        # feature 2
        moments = cv2.HuMoments(cv2.moments((img).astype(np.uint8))).flatten()
        # print(m)
        # mu = moments_central(1-img)
        # nu = moments_normalized(mu)
        # moments = moments_hu(nu)

        # for moment in moments:
        # print(len(moments_hu(nu)))
        # print(len(moments))
        for moment in moments:
            features.append(moment)
        # feature 3
        regions = region16(img)
        for region in regions:
            volume = regionVolume(region)
            features.append(volume)
        # print("features:",len(features))
    return features


def calculateDistance(x1, x2):
    distance = np.linalg.norm(x1 - x2)
    return distance


def read_training_image(path):
    gray = rgb2gray(io.imread(path))
    # gray = gaussian(gray,1)
    thresh = threshold_sauvola(gray, window_size=61)
    normalize = gray > thresh
    return normalize


def KNN(test_point, training_features, y_train, k):
    classification = 0

    minDist = [999999 for i in range(k)]
    minClass = [3 for i in range(k)]

    features_triple_eighth_down = training_features[y_train == 0]
    features_double_eighth_down = training_features[y_train == 1]
    features_double_sixteenth_down = training_features[y_train == 2]
    features_quadruple_sixteenth_down = training_features[y_train == 3]
    features_clef = training_features[y_train == 4]
    features_double_flat = training_features[y_train == 5]
    features_double_sharp = training_features[y_train == 6]
    features_flat = training_features[y_train == 7]
    features_half_note = training_features[y_train == 8]
    features_quarter_note = training_features[y_train == 9]
    features_eighth_note = training_features[y_train == 10]
    features_sharp = training_features[y_train == 11]
    # features_natural = training_features[y_train==12]
    features_whole_note = training_features[y_train == 12]
    features_sixteenth_note = training_features[y_train == 13]
    features_32th_note = training_features[y_train == 14]
    # features_bar_line = training_features[y_train==15]
    # features_triple_sixteenth = training_features[y_train==15]
    features_chord = training_features[y_train == 15]
    features_bar_line = training_features[y_train == 16]
    features_four_four = training_features[y_train == 17]
    features_four_two = training_features[y_train == 18]

    for i in features_triple_eighth_down:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 0
    for i in features_double_eighth_down:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 1
    for i in features_double_sixteenth_down:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 2
    for i in features_quadruple_sixteenth_down:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 3
    for i in features_clef:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 4
    for i in features_double_flat:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 5
    for i in features_double_sharp:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 6
    for i in features_flat:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 7
    for i in features_half_note:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 8
    for i in features_quarter_note:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 9
    for i in features_eighth_note:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 10
    for i in features_sharp:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 11
    for i in features_whole_note:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 12
    for i in features_sixteenth_note:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 13
    for i in features_32th_note:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 14
    for i in features_chord:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 15
    for i in features_bar_line:
        c = calculateDistance(i,test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 16
    for i in features_four_four:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 17
    for i in features_four_two:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 18

    '''
    for i in features_triple_sixteenth:
        c = calculateDistance(i,test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 15
    '''
    '''
    for i in features_natural:
        c = calculateDistance(i,test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 12
    '''

    # ------------------------------------------------------------------------------------------------------

    zero = minClass.count(0)
    one = minClass.count(1)
    two = minClass.count(2)
    three = minClass.count(3)
    four = minClass.count(4)
    five = minClass.count(5)
    six = minClass.count(6)
    seven = minClass.count(7)
    eight = minClass.count(8)
    nine = minClass.count(9)
    ten = minClass.count(10)
    eleven = minClass.count(11)
    twelve = minClass.count(12)
    thirteen = minClass.count(13)
    fourteen = minClass.count(14)
    fifteen = minClass.count(15)
    sixteen = minClass.count(16)
    seventeen = minClass.count(17)
    eighteen = minClass.count(18)

    temp = [zero, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen,
            fifteen, sixteen, seventeen, eighteen]
    classification = temp.index(max(temp))
    return classification


def calc_accuracy(knns, true_values, ntest):
    total_predictions = ntest  # np.array(test_images).shape[0]
    correct_knn = 0
    for i in range(len(true_values)):
        if true_values[i] == knns[i]:
            correct_knn += 1
    accuracy_knn = correct_knn / len(true_values)
    print("K-Nearest Neighbour Classifier Accuracy: ", accuracy_knn, "%")
    return accuracy_knn


def predict(test_images, shapes, true_values, training_features, y_train):
    knns = []
    for i in range(len(test_images)):
        # Read each image in the test directory, preprocess it and extract its features.
        img_original = read_training_image(test_images[i])
        components = find_regionprop(img_original)
        test_point = extract_features(components)

        # Print the actual class of each test figure.
        print("Actual class :", shapes[true_values[i]])
        print("---------------------------------------")

        k = 3
        knn_prediction = KNN(test_point, training_features, y_train, k)
        knns.append(knn_prediction)

        print("K-Nearest Neighbours Prediction          :", shapes[knn_prediction])
        print("===========================================================================")
        '''
        # Visualize each test figure.
        fig = plt.figure()
        plt.imshow(img_original)
        plt.axis("off")
        plt.show()
        '''

    return knns


def single_note_pitch(bbox, first_staff, h, s, topbot):
    if topbot == 1:
        if bbox[2] - (s / 2) >= first_staff + (5 * h) + (5 * s) - 0.2 * s:
            print("c")
            return "c"
        elif bbox[2] >= first_staff + (5 * h) + (5 * s):
            print("d")
            return "d"
        elif bbox[2] - (s / 2) >= first_staff + (4 * h) + (4 * s) - 0.2 * s:
            print("e")
            return "e"
        elif bbox[2] >= first_staff + (4 * h) + (4 * s):
            print("f")
            return "f"
        elif bbox[2] - (s / 2) >= first_staff + (3 * h) + (3 * s) - 0.2 * s:
            print("g")
            return "g"
        elif bbox[2] >= first_staff + (3 * h) + (3 * s):
            print("a")
            return "a"
        elif bbox[2] - (s / 2) >= first_staff + (2 * h) + (2 * s) - 0.2 * s:
            print("b")
            return "b"
    else:
        if bbox[0] <= first_staff - h - (2 * s):
            print("b2")
            return "b2"
        elif bbox[0] + (s / 2) <= first_staff - s + 0.2 * s:
            print("a2")
            return "a2"
        elif bbox[0] <= first_staff - s:
            print("g2")
            return "g2"
        elif bbox[0] + (s / 2) <= first_staff + h + 0.2 * s:
            print("f2")
            return "f2"
        elif bbox[0] <= first_staff + h:
            print("e2")
            return "e2"
        elif bbox[0] + (s / 2) <= first_staff + (2 * h) + s + 0.2 * s:
            print("d2")
            return "d2"
        elif bbox[0] <= first_staff + (2 * h) + s:
            print("c2")
            return "c2"
        elif bbox[0] + (s / 2) <= first_staff + (2 * h) + (2 * s) + 0.2 * s:
            print("b")
            return "b"



def getfirst_staff_line(img):
    rows = img.shape[0]
    cols = img.shape[1]
    h = np.sum(img == 0, 1)
    staff = h > 0.59 * cols
    return np.argwhere(staff)



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

'''
# Classification 3la matofrag 

# TODO Classification FINDING THE RHYTHM OF THE NOTES AND THE NUMBER OF THE NOTES //KARIM
##### PROCESSING EACH NOTE SEGMENT
# show_images(notesImages)
image = notesImages[21]  # [22]  # [17] # 28 # 10 # 6 # 21

show_images([notesImages[21]])  # 4 #1 #14# 26 27 39 25

avgAreas = np.average(componentsAreas(notes))
ratio = notesImages[21].shape[0] / notesImages[21].shape[1]
area = notes[21].area
print("Area: ", area, " Average area:", avgAreas)
print("ratio: ", ratio)
if ratio > 1.4:
    print("Accidentals or single stem")
    # TODO ACCIDENTAL CLASSIFICATION
    quarter_one = image[0:image.shape[0]//2, 0: image.shape[1]//2]
    quarter_two = image[0:image.shape[0]//2, image.shape[1]//2:image.shape[1]]
    quarter_three = image[image.shape[0]//2:image.shape[0], 0: image.shape[1]//2]
    quarter_four = image[image.shape[0]//2:image.shape[0], image.shape[1]//2:image.shape[1]]


    #### Removing Stems to count the number of notes
    V_staff_indices = find_verticalLines(image)
    print(V_staff_indices)
    # image[:, V_staff_indices] = 0
    stems_indices, stem_count = countStems(V_staff_indices)

    # s, t = get_references(image)
    image = binary_opening(image, np.ones((1, t + 4)))
    show_images([image])
    #### Find out where the notes are top or bottom
    top_bottom, top_image, bot_image = classifyNotePositionInSegment(image)
    print("Top or bottom:", top_bottom)
    print(stems_indices)

    image_copy = (image.copy())
    image_filled_hole = image_copy

    # TODO FIND WHETHER IT WAS FILLED OR HOLLOW THEN GO BACK TO USING IMAGE_COPY

    ##### Find Row Histogram
    row_histogram = np.array([sum(image_filled_hole[i, :]) for i in range(image_filled_hole.shape[0])])
    print(row_histogram.shape)

    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
    note_threshold = image.shape[1] // 2 - 1
    peaks, _ = find_peaks(row_histogram, height=note_threshold)

    ##### Plot Peaks on histogram
    print("peaks", peaks)
    plt.plot(row_histogram)
    plt.plot(peaks, row_histogram[peaks], "x")
    plt.plot(np.zeros_like(row_histogram), "--", color="gray")
    plt.show()

    ##### Find the local Minimas between the number of notes
    stacced_flag = 0
    numberOfPeaks = len(peaks)
    # localMinimas = []
    if numberOfPeaks == 1:
        print("One Note i.e no Chord")
    elif numberOfPeaks == 2:
        print("Two Notes Chord")

        ##### Find the distance between tail peak and note peak to differentiate between // IN CASE OF FILLING HOLES
        if peaks[0] - peaks[1] > peaks[1] // 2 or peaks[1] - peaks[0] > peaks[1] // 2:
            print("False Identification of tail as a note")
            numberOfPeaks -= 1
            if max(row_histogram[peaks[0]], row_histogram[peaks[1]]) == row_histogram[peaks[0]]:
                peaks = [peaks[0]]
            else:
                peaks = [peaks[1]]
            print("peaks", peaks)
        else:
            # for detecting stacced notes, we will put a threshold on the peaks and increment them accordingly
            # print(row_histogram[peaks])
            if row_histogram[peaks[0]] > row_histogram[peaks[1]] + row_histogram[peaks[1]] // 2 or \
                    row_histogram[peaks[0]] + row_histogram[peaks[0]] // 2 < row_histogram[peaks[1]]:
                print("Three notes Chord Stacced!")
                # TODO WE NEED TO FIND THE LOCALMINIMAS FOR SEGMENTATION FOR HOLLOW/ RIGID SPHERE
                # Stacced flag for number of peaks increment
                stacced_flag = 1
                numberOfPeaks += 1
                # peaks = [np.min(peaks), np.max(peaks//2), np.max(peaks//2)]
                # peaks.append(np.max(peaks//2))
            # else:
            #    localMinimas.append((peaks[0] + peaks[1]) // 2)
    elif numberOfPeaks == 3:
        print("Three Notes Chord")
        # localMinimas.append((peaks[0] + peaks[1]) // 2)
        # localMinimas.append((peaks[1] + peaks[2]) // 2)
else:
    print("Beamed notes")

if stem_count == 0:
    print("One Whole note")
elif stem_count == 1:

    print("Many notes i.e: Chord or one single note( half or quarter or eighth or sixteenth)")

'''
