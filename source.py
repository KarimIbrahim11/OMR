import imutils as imutils
from cv2.cv2 import CV_32F

# from OMR.util import *
from scipy.ndimage import binary_fill_holes
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

image = gray.copy()

# TODO SKEWNESS NEDIF AGAINST CAPTURED PERSPECTIVE // AMIR

rotated, gray = deskew(gray)
# show_images([gray], ["Gray: "])
# show_images([rotated], ["binary"])

# TODO STAFF LINE REMOVAL
rotated_copy = rotated.copy()
# withoutLines = removeHLines(rotated_copy)
# show_images([rotated, withoutLines], ["Binary", " After Line Removal"])

# Remove Staff TIFA
# staff_indices = find_stafflines(rotated, 0, 0)
# print(staff_indices)
# rotated[staff_indices, :] = 0
s, t = get_references(rotated)
withoutLines = binary_opening(rotated, np.ones((t + 2, 1)))
# Added another opening for noise removal:
# withoutLines = binary_opening(withoutLines, np.ones((3, 3)))
show_images([rotated, withoutLines], ["Rotated", "After Line Removal"])

# TODO replace the 6 with a variable dependant on the ratio between The width and the height of the image And remove
#  the non uniform closing

# Non uniform Closing
# First dilate if there's a horizontal skip
withoutLines_dilated = withoutLines
selem = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]])
withoutLines_dilated = binary_dilation(withoutLines, selem)
# Second Erode for vertical segmentations
selem = np.array([[0, 0, 1, 0] * 3]).reshape((4, 3))
withoutLines_dilated = binary_erosion(withoutLines_dilated, selem)
withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((6, 1)))

show_images([withoutLines_dilated], ["Dilated"])

# withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((5, 5)))
# withoutLines_dilated = binary_opening(withoutLines_dilated, np.ones((5, 5)))
# io.imsave('savedImage2.png', withoutLines_dilated)
# path = 'savedImage.png'
# rtt = read_image(path)
# withoutLines_dilated = withoutLines

# regionproprs object
notes, notesImages = CCA(withoutLines_dilated)

print([notes[0]])
show_images(notesImages)
boxes = RetrieveComponentBox(notes)
binary_notes_with_lines = segmentBoxesInImage(boxes, rotated)
gray_notes_with_lines = segmentBoxesInImage(boxes, gray)
# notesImages = componentsToImages(notes)
displayComponents(withoutLines_dilated, notes)

# TODO TEMPALTE MATCH THE CLEFS //JOE
'''
clef_template = read_image('clef.jpg')
clef_template = resize(clef_template, (clef_template.shape[0] // 10, clef_template.shape[1] // 10))
grayyyyyyyy = rgb2gray(clef_template)
print(grayyyyyyyy.shape)
binaryclef = grayyyyyyyy > threshold_otsu(grayyyyyyyy)
print(binaryclef.shape, notesImages[1].shape)
template_Match(notesImages[24], binaryclef)
'''

# TODO AMIR: FIND PITCH
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

# TODO Classification FINDING THE RHYTHM OF THE NOTES AND THE NUMBER OF THE NOTES //KARIM
##### PROCESSING EACH NOTE SEGMENT
# show_images(notesImages)
image = notesImages[52]  # [22]  # [17] # 28

#### FINDING POSITION OF NOTE HEAD --> TOP = 0, BOTTOM = 1 and returning segmented images
show_images([notesImages[55]])  # 4 #1 #14# 26 27 39 25

avgAreas = np.average(componentsAreas(notes))
ratio = notesImages[55].shape[0] / notesImages[55].shape[1]
area = notes[55].area
print("Area: ", area, " Average area:", avgAreas)
print("ratio: ", ratio)
if ratio > 1.4:
    print("Accidentals or single stem")
    # TODO ACCIDENTAL CLASSIFICATION
else:
    print("Beamed notes")

#### Removing Stems to count the number of notes
V_staff_indices = find_verticalLines(image)
print(V_staff_indices)
image[:, V_staff_indices] = 0

top_bottom, top_image, bot_image = classifyNotePositionInSegment(notesImages[5])
print("Top or bottom:", top_bottom)
stems_indices, stem_count = countStems(V_staff_indices)
print(stems_indices)
if stem_count == 0:
    print("One Whole note")
elif stem_count == 1:

    # image_filled_hole = binary_closing(image, np.ones((7, 7)))
    # show_images([image, image_filled_hole])
    # top_bottom, top_image, bot_image = classifyNotePositionInSegment(image)

    print("Many notes i.e: Chord or one single note( half or quarter or eighth or sixteenth)")

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

    '''
    ##### Segment the image based on
    # for i in range(len(localMinimas)):

    image = img_as_ubyte(image)
    show_images([image])
    edges = canny(image, sigma=2, low_threshold=10, high_threshold=50)
    show_images([edges])

    # Detect two radii
    hough_radii = 0
    if stacced_flag == 1:
        hough_radii = np.arange(image.shape[1] // 4, image.shape[1] // 4 + 10, 1)
    else:
        hough_radii = np.arange(image.shape[1] // 2 - 10, image.shape[1] // 2, 5)
    hough_res = hough_circle(edges, hough_radii)
    print("hough radii", hough_radii)
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=numberOfPeaks)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()
elif stem_count == 2:
    print("Beamed note, we need to find the number of tails")
'''
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
