from util import *

path = 'cases/02.PNG'
img = read_image(path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(img)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(img))
# gray = resize(gray, (256, 256))
# TODO 7AD YE3MEL LOCAL BINARIZATION NEGARAB
# TODO SKEWNESS NEDIF AGAINST CAPTURED
rotated = deskew(gray)
show_images([rotated], ["binary"])

'''
newshape = np.array(rotated.shape)
print(newshape)

if newshape[0] / newshape[1] > 2.0 or newshape[1] / newshape[0] > 2.0:
    factor = 0.75
    while np.max(newshape) > 500:
        newshape = newshape * factor
        print("anahena ya salama")


#resized_image = skimage.transform.resize(rotated, newshape)
print(resized_image.shape)
show_images([resized_image])
'''
# photo = resize(rotated, (256, 256))
# hist,_ = histogram(photo, nbins=256)

# Remove Staff AMIR
# bin, _ = otsu_binarize(rotated)
# TODO A MORE ROBUST TO SKEWNESS STAFF LINE REMOVAL
# withoutLines = removeHLines(rotated)
# show_images([rotated, withoutLines], ["Binary", "After Line Removal"])

# Remove Staff TIFA
# staff_indices = find_stafflines(rotated, 0, 0)
# print(staff_indices)
# rotated[staff_indices, :] = 0
s, t = get_references(rotated)
withoutLines = binary_opening(rotated, np.ones((t + 2, 1)))
show_images([rotated, withoutLines], ["Rotated", "After Line Removal"])

# new_image = binary_closing(rotated, np.ones((3, 1)))

# photo = resize(new_image, (256, 256))
# hist, _ = histogram(photo, nbins=256)
# showHist(hist)


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

# withoutLines_dilated = binary_erosion(withoutLines_dilated, np.array(selem))
# selem = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
# withoutLines_dilated = binary_erosion(withoutLines_dilated, np.array(selem))
show_images([withoutLines_dilated], ["Dilated"])
notes = CCA(withoutLines_dilated)
displayComponents(gray, notes)
# TODO Thinning each image can help in some features

notesImages = componentsToImages(notes)

# TODO CLASSIFICATION USING SIFT OR A PLAN B
'''
for Image in notesImages:
    Image = thin(Image, 5)
    show_images([Image])
'''
show_images(notesImages)
# show_images([binary_closing(notesImages[2])])
# selem = [[0,0,0], [1, 1, 1], [1,1,1], [0, 0, 0]]
# show_images([binary_erosion(binary_erosion(notesImages[35]))])
# show_images([notesImages[36]])

# new_image = new_image.astype(float)
# icut, iicut = 0, 0
# r, c = new_image.shape
# notesList = []
# notFinished = True
# for k in range(2):
#     flag = True
#     for j in range(icut, c):
#         flag = True
#         for i in range(iicut, r):
#             if new_image[i][j] != 0:
#                 flag = False
#                 break
#         if flag:
#             icut, iicut = j, j+400
#             if icut > 200:
#                 break
#         if np.sum(new_image[:][j]) > 500:
#             notFinished = False
#     print("i, j = " + str(icut) + ", " + str(iicut))
#     n1 = new_image[:, icut:iicut]
#     n1 = resize(n1, (256, 256))
#     notesList.append(n1)
#     show_images([n1])


'''
#def segmentVertical(rimg)
hrimg = np.sum(rimg, 0)
m = len(hrimg)

for i in range(m):
    if hrimg[i] == 0:
        rimg[:, i] = 1
   # return rimg


img_with_boxes, boxes = draw_contours(rimg_dilated)
show_images([rimg_dilated, img_with_boxes])
'''
