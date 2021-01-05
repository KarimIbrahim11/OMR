from OMR.util import *

# if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
path = 'cases/10.PNG'
img = read_image(path)
gray = rgb2gray(img)
rotated = deskew(gray)
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
withoutLines = removeHLines(rotated)

show_images([rotated, withoutLines], ["Binary", "After Line Removal"])

# Remove Staff TIFA
# staff_indices = find_stafflines(rotated, 0, 0)
# print(staff_indices)
# rotated[staff_indices, :] = 0

# new_image = binary_closing(rotated, np.ones((3, 1)))

# photo = resize(new_image, (256, 256))
# hist, _ = histogram(photo, nbins=256)
# showHist(hist)


# Non uniform Closing
# First dilate if there's a horizontal skip
selem = np.array([[1, 1, 1], [0, 0, 0],  [1, 1, 1]])
withoutLines_dilated = binary_dilation(withoutLines, selem)
# Second Erode for vertical segmentations
selem = np.array([[0, 0, 1, 0]*3]).reshape((4, 3))
#selem = np.array([[0, 0, 1, 0],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 0]])
withoutLines_dilated = binary_erosion(withoutLines_dilated, selem)
withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((7, 1)))
withoutLines_dilated = binary_closing(withoutLines_dilated)

# withoutLines_dilated = binary_erosion(withoutLines_dilated, np.array(selem))
# selem = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
# withoutLines_dilated = binary_erosion(withoutLines_dilated, np.array(selem))
show_images([withoutLines_dilated], ["Dilated"])
notes = CCA(withoutLines_dilated)
displayComponents(withoutLines_dilated, notes)
notesImages = componentsToImages(notes)

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
