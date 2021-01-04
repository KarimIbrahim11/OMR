from util import *

path = 'cases/27_!.jpg'


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


def deskew(image):
    # image = imread( filename, as_grey=True)
    # threshold to get rid of extraneous noise
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
    return rotated


# if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
path = 'cases/08.PNG'
img = read_image(path)
gray = rgb2gray(img)
rotated = deskew(gray)


# gray = median(gray)
# binImg = otsu_binarize(rotated)
# edges = canny(binImg)
# show_images([gray, edges],['image','edges'])
# binImg = binary_closing(binImg)
# image_lines(gray, thres=90)

# photo = resize(rotated, (256, 256))
# hist,_ = histogram(photo, nbins=256)
# showHist(hist)
bin, _ = otsu_binarize(rotated)
removeLines(bin)

# Remove Staff
staff_indices = find_stafflines(rotated, 0, 0)
print(staff_indices)
rotated[staff_indices, :] = 0
new_image = binary_closing(rotated, np.ones((3, 1)))
# Draw Contours
img_with_boxes, boxes = draw_contours(new_image)
show_images([new_image, img_with_boxes])

photo = resize(new_image, (256, 256))
hist, _ = histogram(photo, nbins=256)
# showHist(hist)

