from OMR.util import *

path = 'cases/01.PNG'
img = read_image(path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(img)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(img))




bin,_ = otsu_binarize(gray)
line = bin
m, n = line.shape
for i in range(n):
    c = 0
    for j in range(m):
        if line[j][i] < 1:
            c += 1
    if c >= 5 and c <= 15:
        line[:, i] = 1

show_images([bin, line])
