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

