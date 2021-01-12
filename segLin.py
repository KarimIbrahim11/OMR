#######################
# Draft
######################
from OMR.util import *

path = 'cases/32.jpg'
img = read_image(path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(img)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(img))
#
# thresh = cv2.threshold(gray, 0, 255,
#                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(gray > 0))
angle = cv2.minAreaRect(coords)[-1]
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
    angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make
# it positive
else:
    angle = -angle

(h, w) = gray.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D((w/2, h/2), 270, 1)
rotated = cv2.warpAffine(gray, M, (w, h))
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
show_images([gray, rotated])

# bin, _ = otsu_binarize(gray)
# line = bin
# m, n = line.shape
# for i in range(n):
#     c = 0
#     for j in range(m):
#         if line[j][i] < 1:
#             c += 1
#     if c >= 5 and c <= 10:
#         line[:, i] = 1
#
# show_images([line])
# separatedNotes = []
# for i in range(n):
#     j = 0
#     if line[:, i].all() == 1:
#         while line[:, j].all() == 1:
#             j += 1
#         d = j + i // 2
#         k = 0
#         while line[:, j].any() != 1:
#             k += 1
#         separatedNotes.append(line[:, d: k])
#
# separatedNotes.append(line[:, 45: 90])
# show_images([separatedNotes[0]])
