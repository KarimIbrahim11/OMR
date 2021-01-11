import imutils as imutils
from cv2 import CV_32F

# from OMR.util import *
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks
from skimage import color
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse

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
# show_images([gray], ["Gray: "])
# show_images([rotated], ["binary"])

# TODO STAFF LINE REMOVAL
rotated_copy = rotated.copy()
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

show_images([withoutLines_dilated], ["Dilated"])

# withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((5, 5)))
# withoutLines_dilated = binary_opening(withoutLines_dilated, np.ones((5, 5)))
# io.imsave('savedImage2.png', withoutLines_dilated)
# path = 'savedImage.png'
# rtt = read_image(path)
# withoutLines_dilated = withoutLines

# regionproprs object
notes, notesImages, boxes = CCA(withoutLines_dilated)

print([notes[0]])
show_images(notesImages)
# boxes = RetrieveComponentBox(notesImages)
binary_notes_with_lines = segmentBoxesInImage(boxes, rotated)
gray_notes_with_lines = segmentBoxesInImage(boxes, gray)
show_images(binary_notes_with_lines)
# notesImages = componentsToImages(notes)
displayComponents(withoutLines_dilated, notes)

# TODO TEMPALTE MATCH THE CLEFS //JOE
'''
clef_template = read_image('clef.jpg')
clef_template = resize(clef_template, (clef_template.shape[0] // 10, clef_template.shape[1] // 10))
grayyyyyyyy = rgb2gray(clef_template)
print(grayyyyyyyy.shape)
binaryclef = grayyyyyyyy > threshold_otsu(grayyyyyyyy)
print(binaryclef.shape, notesImages[1].shape)
template_Match(notesImages[24], binaryclef)
'''

# TODO AMIR: FIND PITCH
'''
num_lines = 0
num_lines_list = []
for img in binary_notes_with_lines:
    lm, no = img.shape
    num_lines = 0
    for i in range(1, lm):
        if img[i][1] == 0:
            continue
        elif img[i][1] == 1 and img[i - 1][1] == 0:
            num_lines += 1

    # print(num_lines)
    num_lines_list.append(num_lines)

    if num_lines == 3:
        print("g")
    elif num_lines == 4:
        print("d")
    else:
        print("none")
    # show_images([img])
'''
# TODO Classification FINDING THE RHYTHM OF THE NOTES AND THE NUMBER OF THE NOTES //KARIM
##### PROCESSING EACH NOTE SEGMENT
# show_images(notesImages)
image = notesImages[21]  # [22]  # [17] # 28 # 10 # 6 # 21


show_images([notesImages[21]])  # 4 #1 #14# 26 27 39 25

avgAreas = np.average(componentsAreas(notes))
ratio = notesImages[21].shape[0] / notesImages[21].shape[1]
area = notes[21].area
print("Area: ", area, " Average area:", avgAreas)
print("ratio: ", ratio)
if ratio > 1.4:
    print("Accidentals or single stem")
    # TODO ACCIDENTAL CLASSIFICATION
else:
    print("Beamed notes")

#### Removing Stems to count the number of notes
V_staff_indices = find_verticalLines(image)
print(V_staff_indices)
#image[:, V_staff_indices] = 0
stems_indices, stem_count = countStems(V_staff_indices)


# s, t = get_references(image)
image = binary_opening(image, np.ones((1, t + 4)))
show_images([image])
#### Find out where the notes are top or bottom
top_bottom, top_image, bot_image = classifyNotePositionInSegment(image)
print("Top or bottom:", top_bottom)
print(stems_indices)
if stem_count == 0:
    print("One Whole note")
elif stem_count == 1:

    # image_filled_hole = binary_closing(image, np.ones((7, 7)))
    # show_images([image, image_filled_hole])
    # top_bottom, top_image, bot_image = classifyNotePositionInSegment(image)

    print("Many notes i.e: Chord or one single note( half or quarter or eighth or sixteenth)")

    image_copy = (image.copy())
    image_filled_hole = image_copy

    # TODO FIND WHETHER IT WAS FILLED OR HOLLOW THEN GO BACK TO USING IMAGE_COPY

    ##### Find Row Histogram
    row_histogram = np.array([sum(image_filled_hole[i, :]) for i in range(image_filled_hole.shape[0])])
    print(row_histogram.shape)

    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
    note_threshold = image.shape[1] // 2 - 1
    peaks, _ = find_peaks(row_histogram, height=note_threshold)

    ##### Plot Peaks on histogram
    print("peaks", peaks)
    plt.plot(row_histogram)
    plt.plot(peaks, row_histogram[peaks], "x")
    plt.plot(np.zeros_like(row_histogram), "--", color="gray")
    plt.show()

    ##### Find the local Minimas between the number of notes
    stacced_flag = 0
    numberOfPeaks = len(peaks)
    # localMinimas = []
    if numberOfPeaks == 1:
        print("One Note i.e no Chord")
    elif numberOfPeaks == 2:
        print("Two Notes Chord")

        ##### Find the distance between tail peak and note peak to differentiate between // IN CASE OF FILLING HOLES
        if peaks[0] - peaks[1] > peaks[1] // 2 or peaks[1] - peaks[0] > peaks[1] // 2:
            print("False Identification of tail as a note")
            numberOfPeaks -= 1
            if max(row_histogram[peaks[0]], row_histogram[peaks[1]]) == row_histogram[peaks[0]]:
                peaks = [peaks[0]]
            else:
                peaks = [peaks[1]]
            print("peaks", peaks)
        else:
            # for detecting stacced notes, we will put a threshold on the peaks and increment them accordingly
            # print(row_histogram[peaks])
            if row_histogram[peaks[0]] > row_histogram[peaks[1]] + row_histogram[peaks[1]] // 2 or \
                    row_histogram[peaks[0]] + row_histogram[peaks[0]] // 2 < row_histogram[peaks[1]]:
                print("Three notes Chord Stacced!")
                # TODO WE NEED TO FIND THE LOCALMINIMAS FOR SEGMENTATION FOR HOLLOW/ RIGID SPHERE
                # Stacced flag for number of peaks increment
                stacced_flag = 1
                numberOfPeaks += 1
                # peaks = [np.min(peaks), np.max(peaks//2), np.max(peaks//2)]
                # peaks.append(np.max(peaks//2))
            # else:
            #    localMinimas.append((peaks[0] + peaks[1]) // 2)
    elif numberOfPeaks == 3:
        print("Three Notes Chord")
        # localMinimas.append((peaks[0] + peaks[1]) // 2)
        # localMinimas.append((peaks[1] + peaks[2]) // 2)

