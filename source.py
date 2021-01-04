from util import *

path = 'cases/27_!.jpg'



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
rimg = removeLines(bin)

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

new_image = new_image.astype(float)
icut, iicut = 0, 0
r, c = new_image.shape
notesList = []
notFinished = True
for k in range(2):
    flag = True
    for j in range(icut, c):
        flag = True
        for i in range(iicut, r):
            if new_image[i][j] != 0:
                flag = False
                break
        if flag:
            icut, iicut = j, j+400
            if icut > 200:
                break
        if np.sum(new_image[:][j]) > 500:
            notFinished = False
    print("i, j = " + str(icut) + ", " + str(iicut))
    n1 = new_image[:, icut:iicut]
    n1 = resize(n1, (256, 256))
    notesList.append(n1)
    show_images([n1])

hrimg = np.sum(rimg, 0)
m = len(hrimg)


for i in range(m):
    if hrimg[i] == 0:
        rimg[:, i] = 1

show_images([rimg])