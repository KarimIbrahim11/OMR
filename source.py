from cv2.cv2 import CV_32F

# from OMR.util import *
from scipy.signal import find_peaks
from skimage import color
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse

from util import *

path = 'cases/01.PNG'
The_image = read_image(path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(The_image)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(The_image))
# negativeandSave('quarternote.png')
# TODO 7AD YE3MEL LOCAL BINARIZATION NEGARAB
# TODO SKEWNESS NEDIF AGAINST CAPTURED
rotated, gray = deskew(gray)

# show_images([gray], ["Gray: "])
# show_images([rotated], ["binary"])

# Remove Staff AMIR
# bin, _ = otsu_binarize(rotated)
# TODO A MORE ROBUST TO SKEWNESS STAFF LINE REMOVAL
rotated_copy = rotated.copy()
withoutLines = removeHLines(rotated_copy)
# show_images([rotated, withoutLines], ["Binary", " After Line Removal"])

# Remove Staff TIFA
# staff_indices = find_stafflines(rotated, 0, 0)
# print(staff_indices)
# rotated[staff_indices, :] = 0
# s, t = get_references(rotated)
# withoutLines = binary_opening(rotated, np.ones((t+2, 1)))
# show_images([rotated, withoutLines], ["Rotated", "After Line Removal"])

# Non uniform Closing
# First dilate if there's a horizontal skip
withoutLines_dilated = withoutLines

# TODO replace the 6 with a variable dependant on the ratio between The width and the height of the image And remove
#  the non uniform closing
selem = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]])
withoutLines_dilated = binary_dilation(withoutLines, selem)
# Second Erode for vertical segmentations
selem = np.array([[0, 0, 1, 0] * 3]).reshape((4, 3))
withoutLines_dilated = binary_erosion(withoutLines_dilated, selem)
withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((6, 1)))
# io.imsave('savedImage2.png', withoutLines_dilated)
# path = 'savedImage.png'
# rtt = read_image(path)
# withoutLines_dilated = withoutLines
show_images([withoutLines_dilated], ["Dilated"])
# regionproprs object
notes = CCA(withoutLines_dilated)
# TODO bounding Boxes for each component
boxes = RetrieveComponentBox(notes)
# print(boxes)

binary_notes_with_lines = segmentBoxesInImage(boxes, rotated)
gray_notes_with_lines = segmentBoxesInImage(boxes, gray)
# TODO AMIR: NOTES WITH LINES FEL ARRAY DA:
# show_images([rotated], ["BOUNDING BOXES"])
# show_images(binary_notes_with_lines)
displayComponents(withoutLines_dilated, notes)
notesImages = componentsToImages(notes)

# notes_hp = []
# for img in binary_notes_with_lines:
# img[:, 0] = 0
# hproj = np.sum(img, 1)
# m = np.max(hproj)
# w = img.shape[1]
# result = np.zeros((hproj.shape[0], w))
# # Draw a line for each row
# for row in range(img.shape[0]):
#     cv2.line(result, (0, row), (int(hproj[row] * w / m), row), (255, 255, 255), 1)
# notes_hp.append((img, result))


num_lines = 0
num_lines_list = []
for img in binary_notes_with_lines:
    lm, no = img.shape
    num_lines = 0
    for i in range(1, lm):
        if img[i][1] == 0:
            continue
        elif img[i][1] == 1 and img[i - 1][1] == 0:
            num_lines += 1

    # print(num_lines)
    num_lines_list.append(num_lines)

    if num_lines == 3:
        print("g")
    elif num_lines == 4:
        print("d")
    else:
        print("none")
    # show_images([img])

# show_images(notesImages)
show_images(notesImages)
# TODO FINDING THE RHYTHM OF THE NOTES AND THE NUMBER OF THE NOTES
##### Remove Vertical Stems
image = notesImages[13]
V_staff_indices = find_verticalLines(image)
image[:, V_staff_indices] = 0

##### Find Row Histogram
row_histogram = np.array([sum(image[i, :]) for i in range(image.shape[0])])
print(row_histogram.shape)

##### Find Peaks corresponding to each note with the threshold
note_threshold = image.shape[1]//2
peaks, _ = find_peaks(row_histogram, height=note_threshold)

##### Plot Peaks on histogram
# print(peaks)
plt.plot(row_histogram)
plt.plot(peaks, row_histogram[peaks], "x")
plt.plot(np.zeros_like(row_histogram), "--", color="gray")
plt.show()

##### Find the local Minimas between the number of notes
numberOfPeaks = len(peaks)
localMinimas = []
if numberOfPeaks == 1:
    print("One Note i.e no Chord")
elif numberOfPeaks == 2:
    print("Two Notes Chord")
    localMinimas.append((peaks[0]+peaks[1]) // 2)
elif numberOfPeaks == 3:
    print("Three Notes Chord")
    localMinimas.append((peaks[0]+peaks[1]) // 2)
    localMinimas.append((peaks[1]+peaks[2]) // 2)


##### Segment the image based on
# for i in range(len(localMinimas)):




image = img_as_ubyte(image)
show_images([image])
edges = canny(image, sigma=2, low_threshold=10, high_threshold=50)
show_images([edges])

# Detect two radii
hough_radii = np.arange(image.shape[1]//2-10, image.shape[1]//2, 5)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()

'''
# Load picture and detect edges
referenceImg = gray_notes_with_lines[2].astype(float)
image_gray = notesImages[2].astype(float)
show_images([image_gray])
image_gray *= 255.0
print("Image gray shape : ", image_gray.shape)
edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(edges, accuracy=100, max_size=image_gray.shape[1])
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]

orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
# print("Parameters of ellipse: ",cy, cx)
# np.min(cy):np.max(cy), np.min(cx):np.max(cx)
referenceImg[cy, cx] = 128.0
# The_image[cy, cx, 1] = 0
# The_image[cy, cx, 2] = 255
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(referenceImg)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
'''
'''
for Image in notesImages:
    Image = thin(Image, 5)
    show_images([Image])
'''

# TODO CLASSIFICATION USING SIFT OR A PLAN B
'''
show_images(notesImages)
sampleimg = notesImages[12]
show_images([sampleimg])
print(sampleimg.shape)
shape = np.asarray(sampleimg.shape)
shape = np.append(shape, 3)
print(shape)
wawa = np.zeros(shape)
print("wawa:", wawa.shape)
print(np.max(sampleimg))
wawa[:, :, 0] = np.array(sampleimg * 255, dtype=np.uint8)
wawa[:, :, 1] = np.array(sampleimg * 255, dtype=np.uint8)
wawa[:, :, 2] = np.array(sampleimg * 255, dtype=np.uint8)

wawa = np.negative(wawa)
wawa = cv2.normalize(wawa, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
print(wawa)


#img = np.zeros((int(sampleimg), 3))

#img[:, :, 0] = sampleimg*255.0/255.0
#img[:, :, 1] = sampleimg*255.0/255.0
#img[:, :, 2] = sampleimg*255.0/255.0
#print(img)


img1 = cv2.imread('quarternote2.jpg')

scale_percent = 100 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
print("yasta:", img1.shape)
# img2 = cv2.imread(path)

# convert images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

########################
# wawa = cv2.imread('cases/02.PNG')
########################
wawa = cv2.cvtColor(wawa, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(rotated)
# create SIFT object
sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT features in both images
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(wawa, None)

# des1 = des1.astype('float32')
# des2 = des2.astype('float32')
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
MIN_MATCH_COUNT = 10
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
        good.append(m)
print(np.sum(matchesMask))

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    print(M)
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)

    wawa = cv2.polylines(wawa, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, wawa, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()
'''
# show_images(notesImages)
