#import imutils as imutils
from cv2 import CV_32F

# from OMR.util import *
from scipy.signal import find_peaks

from util import *

path = 'cases/02.PNG'
The_image = read_image(path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(The_image)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(The_image))

image = gray.copy()

# TODO SKEWNESS NEDIF AGAINST CAPTURED PERSPECTIVE // AMIR

rotated, gray = deskew(gray)
rotated_copy = rotated.copy()
# show_images([gray], ["Gray: "])
# show_images([rotated], ["binary"])

# TODO STAFF LINE REMOVAL

imagesNotes=split_images(rotated_copy, 1)
show_images([imagesNotes[0]],["sora"])
rotated = imagesNotes[0]
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

# show_images([withoutLines_dilated], ["Dilated"])

# withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((5, 5)))
# withoutLines_dilated = binary_opening(withoutLines_dilated, np.ones((5, 5)))
# io.imsave('savedImage2.png', withoutLines_dilated)
# path = 'savedImage.png'
# rtt = read_image(path)
# withoutLines_dilated = withoutLines

notes, notesImages, boxes = CCA(withoutLines_dilated)
show_images(notesImages)
# boxes = RetrieveComponentBox(notesImages)
binary_notes_with_lines = segmentBoxesInImage(boxes, rotated)
gray_notes_with_lines = segmentBoxesInImage(boxes, gray)
show_images(binary_notes_with_lines)

# for image in notesImages:
    # Classification Comes next
# notesImages = componentsToImages(notes)
# displayComponents(withoutLines_dilated, notes)

