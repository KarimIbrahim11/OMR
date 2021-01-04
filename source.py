from util import *
# if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
path = 'cases/27_!.jpg'
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