################## SIFT USING OPENCV
'''
img1 = cv2.imread('trebleclef.jpg')

img2 = cv2.imread(path)

# convert images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(rotated)
# create SIFT object
sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT features in both images
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


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

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()
'''



# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
#
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#
#
# plt.imshow(img3,),plt.show()

####################################### INITAL PRE PROCESSING ATTEMPTS
# gray = median(gray)
# binImg = otsu_binarize(rotated)
# edges = canny(binImg)
# show_images([gray, edges],['image','edges'])
# binImg = binary_closing(binImg)

# image_lines(gray, thres=90)


##################################### CCA APPROACH UNTESTED
'''
# Perform CCA on the mask
labeled_image = skimage.measure.label(rotated, connectivity=2, return_num=True)
NoOfComponents= labeled_image[1]


#image_label_overlay = label2rgb(labeled_image[0], image=image, bg_label=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)#image_label_overlay)


x=skimage.measure.regionprops(labeled_image[0])

#print (x)

for region in x:
    # take regions with large enough areas
    if region.area >= 2000:
        print("in")
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()

'''
################################### DRAW LINES USED IN HOUGH SPACE
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

############### FINDING PEAKS TOP AND BOT IMAGE

'''
row_histogram_not_filled = np.array([sum(image_filled_hole[i, :]) for i in range(image_filled_hole.shape[0])])
print(row_histogram.shape)

##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
note_threshold = image.shape[1] // 2 - 1
peaks_not_filled_holes, _ = find_peaks(row_histogram_not_filled, height=note_threshold)

##### Plot Peaks on histogram
print("peaks", peaks_not_filled_holes)
plt.plot(row_histogram_not_filled)
plt.plot(peaks_not_filled_holes, row_histogram_not_filled[peaks_not_filled_holes], "x")
plt.plot(np.zeros_like(row_histogram_not_filled), "--", color="gray")
plt.show()

##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
note_threshold = image.shape[1] // 2 - 1
peaks_filled_holes, _ = find_peaks(row_histogram_filled, height=note_threshold)

##### Plot Peaks on histogram
print("peaks", peaks_filled_holes)
plt.plot(row_histogram_filled)
plt.plot(peaks_filled_holes, row_histogram_filled[peaks_filled_holes], "x")
plt.plot(np.zeros_like(row_histogram_filled), "--", color="gray")
plt.show()

row_histogram_not_filled = np.array([sum(image_filled_hole[i, :]) for i in range(image_filled_hole.shape[0])])
print(row_histogram_filled.shape)

##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
note_threshold = image.shape[1] // 2 - 1
peaks_not_filled_holes, _ = find_peaks(row_histogram_not_filled, height=note_threshold)

##### Plot Peaks on histogram
print("peaks", peaks_not_filled_holes)
plt.plot(row_histogram_not_filled)
plt.plot(peaks_not_filled_holes, row_histogram_not_filled[peaks_not_filled_holes], "x")
plt.plot(np.zeros_like(row_histogram_not_filled), "--", color="gray")
plt.show()

peaks = []
row_histogram = []
if len(peaks_filled_holes) < len(peaks_not_filled_holes):
    peaks = peaks_not_filled_holes
    row_histogram = row_histogram_not_filled
else:
    peaks = peaks_filled_holes
    row_histogram = row_histogram_filled
'''

############# classificationbayez
'''
if area < 0.6 * avgAreas:
    print("Accidentals/ Rests")
elif ratio >= 1.5 and area >= 0.6 * avgAreas:  # and area > avgAreas - 20:
    print("Single stem")
elif ratio < 1.5 and area >= 0.9 * avgAreas:
    print("Beamed Notes")
'''
'''
#### Calculate ratio of width to height to classify:
avgAreas = np.average(componentsAreas(notes))
print("Average area:", avgAreas)
for i in range(len(notes)):
    # Rows to columns:
    ratio = notesImages[i].shape[0] // notesImages[i].shape[1]
    area = notes[i].area
    if ratio > 1.2:  # and area > avgAreas - 20:
        print("Single stem")
    else:
        print("Beamed note or accidental")
        if area > avgAreas - 10:
            print("Beamed Notes")
        else:
            print("Accidental, rest or dot")
'''
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
