# import imutils as imutils
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

imagesNotes = split_images(rotated_copy, 1)
show_images([imagesNotes[0]])

shapes = ['triple_eighth_down', 'double_eighth_down', 'double_sixteenth_down', 'quadruple_sixteenth_down', "Clef",
          "double_flats", "double_sharps", "flat", "half_note", "quarter_note", "single_eighth note",
          "sharp", "whole_note", "single_sixteenth_note", "single_32th_note", "chord"]
# , "triple_sixteenth", "bar_line",   "single_quarter_note_head up"]

x_train, y_train = training_data(shapes)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape)
print("y_train:", y_train)
number_of_features = 13
training_features = np.zeros((x_train.shape[0], number_of_features))

for i in range(training_features.shape[0]):
    components = find_regionprop(x_train[i])
    # print(y_train[i])
    features = extract_features(components)
    # print(features)
    training_features[i, :] = features

test_images = sorted(glob.glob('KNN Attempt/test/*'))
ntest = len(test_images)

true_values = [1, 1, 2, 3, 0, 4, 5, 6, 7, 8, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15]

knns = predict(test_images, shapes, true_values, training_features, y_train)

accuracy_knn = calc_accuracy(knns, true_values, ntest)

show_images([rotated], ["Binary"])
imagesNotes = split_images(rotated_copy, 1)

for image in imagesNotes:
    show_images([image], ["VIGOOO"])
    # # withoutLines = removeHLines(rotated_copy)
    #
    # # Remove Staff TIFA
    # # staff_indices =find_stafflines(rotated, 0, 0)
    # print(staff_indices)
    # rotated[staff_indices, :] = 0
    s, t = get_references(image)
    withoutLines = binary_opening(image, np.ones((t + 2, 1)))
    # Added another opening for noise removal:
    # withoutLines = binary_opening(withoutLines, np.ones((3, 3)))
    show_images([image, withoutLines], ["Rotated", "After Line Removal"])

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
    # show_images(binary_notes_with_lines)

    for i in range(len(notesImages)):
        show_images([notesImages[i]])
        test_point = extract_features_single_img(notesImages[i], boxes[i])
        classification = KNN(test_point, training_features, y_train, 3)
        print("Classification: ", shapes[classification])

    # for image in notesImages:
    # Classification Comes next
    # notesImages = componentsToImages(notes)
    # displayComponents(withoutLines_dilated, notes)
