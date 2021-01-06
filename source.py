from cv2.cv2 import CV_32F


from OMR.util import *

path = 'cases/02.PNG'
img = read_image(path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(img)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(img))

# TODO 7AD YE3MEL LOCAL BINARIZATION NEGARAB
# TODO SKEWNESS NEDIF AGAINST CAPTURED
rotated = deskew(gray)
show_images([rotated], ["binary"])

# Remove Staff AMIR
# bin, _ = otsu_binarize(rotated)
# TODO A MORE ROBUST TO SKEWNESS STAFF LINE REMOVAL
rotated_copy = rotated.copy()
withoutLines = removeHLines(rotated_copy)
show_images([rotated, withoutLines], ["Binary", "After Line Removal"])

# Remove Staff TIFA
# staff_indices = find_stafflines(rotated, 0, 0)
# print(staff_indices)
# rotated[staff_indices, :] = 0
#s, t = get_references(rotated)
#withoutLines = binary_opening(rotated, np.ones((t+1, 1)))
#show_images([rotated, withoutLines], ["Rotated", "After Line Removal"])

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
#io.imsave('savedImage2.png', withoutLines_dilated)
#path = 'savedImage.png'
#rtt = read_image(path)
show_images([withoutLines_dilated], ["Dilated"])
# regionproprs object
notes = CCA(withoutLines_dilated)
# TODO bounding Boxes for each component
boxes = RetrieveComponentBox(notes)
# print(boxes)
notes_with_lines = []
for box in boxes:
    [Ymin, Xmin, Ymax, Xmax] = box
    #print(box)
    rr, cc = rectangle_perimeter(start = (Ymin,Xmin), end = (Ymax,Xmax), shape=rotated.shape)
    rotated[rr, cc] = 1  # set color white
    notes_with_lines.append(rotated[np.min(rr):np.max(rr), np.min(cc):np.max(cc)])


# TODO AMIR: NOTES WITH LINES FEL ARRAY DA:
notes_with_lines = np.array(notes_with_lines, dtype=object)
show_images([rotated],["BOUNDING BOXES"])
#show_images(notes_with_lines)
displayComponents(withoutLines_dilated, notes)
notesImages = componentsToImages(notes)

# notes_hp = []
# for img in notes_with_lines:
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
for img in notes_with_lines:
    lm, no = img.shape
    num_lines = 0
    for i in range(1, lm):
        if img[i][1] == 0:
            continue
        elif img[i][1] == 1 and img[i-1][1] == 0:
            num_lines += 1



    print(num_lines)
    num_lines_list.append(num_lines)

    if num_lines == 3:
        print("g")
    elif num_lines == 4:
        print("d")
    else:
        print("none")
    show_images([img])


# TODO Thinning each image can help in some features

'''
for Image in notesImages:
    Image = thin(Image, 5)
    show_images([Image])
'''

# TODO CLASSIFICATION USING SIFT OR A PLAN B
'''
show_images(notesImages)
sampleimg = notesImages[37]
show_images([sampleimg])
print(sampleimg.shape)
shape = np.asarray(sampleimg.shape)
shape = np.append(shape,3)
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


img = np.zeros((sampleimg, 3))

img[:, :, 0] = sampleimg*255.0/255.0
img[:, :, 1] = sampleimg*255.0/255.0
img[:, :, 2] = sampleimg*255.0/255.0
print(img)
'''
'''
img1 = cv2.imread('clef.jpg')


scale_percent = 10 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
print("yasta:", img1.shape)
#img2 = cv2.imread(path)

# convert images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
wawa = cv2.cvtColor(wawa, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(rotated)
# create SIFT object
sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT features in both images
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(wawa, None)

des1 = des1.astype('float32')
des2 = des2.astype('float32')
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1, des2, k=2)


# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
MIN_MATCH_COUNT = 10
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1, 0]
        good.append(m)
print(np.sum(matchesMask))

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    print(M)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    dst = cv2.perspectiveTransform(pts,M)

    wawa = cv2.polylines(wawa,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1, kp1, wawa, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()

#show_images(notesImages)
'''
