# import imutils as imutils
from cv2 import CV_32F

# from OMR.util import *
from scipy.signal import find_peaks

from util import *

folder = 'cases/'
path = '01.PNG'
The_image = read_image(folder + path)
if path.lower().endswith('.jpg'):
    gray = rgb2gray(The_image)
elif path.lower().endswith('.png'):
    gray = rgb2gray(rgba2rgb(The_image))

image = gray.copy()

rotated, gray = deskew(gray)
rotated_copy = rotated.copy()

imagesNotes = split_images(rotated_copy, 1)
print([imagesNotes[0].shape[0]])
if [imagesNotes[0].shape[0]] <= [20]:
    [imagesNotes[0]] = [rotated_copy]

show_images(imagesNotes)

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

# show_images([rotated], ["Binary"])
# imagesNotes = split_images(rotated_copy, 1)
clef = False
for image in imagesNotes:
    # image = rotated
    show_images([image], ["VIGOOO"])
    # # withoutLines = removeHLines(rotated_copy)
    #
    # # Remove Staff TIFA
    # # staff_indices =find_stafflines(rotated, 0, 0)
    # print(staff_indices)
    # rotated[staff_indices, :] = 0

    ## space = s, wel height = t
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
    # selem = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]])
    # withoutLines_dilated = binary_dilation(withoutLines, selem)
    # Second Erode for vertical segmentations
    # selem = np.array([[0, 0, 1, 0] * 3]).reshape((4, 3))
    # withoutLines_dilated = binary_erosion(withoutLines_dilated, selem)
    # withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((6, 1)))

    # show_images([withoutLines_dilated], ["Dilated"])

    # withoutLines_dilated = binary_closing(withoutLines_dilated, np.ones((5, 5)))
    # withoutLines_dilated = binary_opening(withoutLines_dilated, np.ones((5, 5)))
    # io.imsave('savedImage2.png', withoutLines_dilated)
    # path = 'savedImage.png'
    # rtt = read_image(path)
    # withoutLines_dilated = withoutLines
    # withoutLines_dilated = binary_erosion(withoutLines_dilated, np.ones((3,3)))
    notes, notesImages, boxes, areas_over_bbox = CCA(withoutLines_dilated)
    show_images(notesImages)

    # boxes = RetrieveComponentBox(notesImages)
    binary_notes_with_lines = segmentBoxesInImage(boxes, rotated)
    gray_notes_with_lines = segmentBoxesInImage(boxes, gray)
    # show_images(binary_notes_with_lines)

    rhythm = ["/8", "/8", "/16", "/16", "clef", "bb", "##", "b", "/2", "/4", "/8", "#", "/1", "/16", "/32", "/4"]
    double_flats = False
    double_sharps = False
    flat_note = False
    sharp_note = False
    for i in range(len(notesImages)):
        show_images([notesImages[i]])
        io.imsave(path + str(i) + '.jpg', 1 - notesImages[i])
        test_point = extract_features_single_img(notesImages[i], areas_over_bbox[i])
        # 'triple_eighth_down', 'double_eighth_down', 'double_sixteenth_down', 'quadruple_sixteenth_down', "Clef",
        #       5    "double_flats", "double_sharps", "flat", "half_note", "quarter_note", "single_eighth note",
        #           "sharp", "whole_note", "single_sixteenth_note", "single_32th_note", "chord"
        classification = KNN(test_point, training_features, y_train, 3)
        print("Classification: ", shapes[classification])
        note_rhythm = rhythm[classification]
        # "double_flats", "double_sharps", "flat", "half_note", "quarter_note", "single_eighth note",
        #           "sharp", "whole_note", "single_sixteenth_note", "single_32th_note", "chord"
        # TODO PITCH
        if classification == 5:  # double flats
            double_flats = True
        elif classification == 6:  # double sharps
            double_sharps = True
        elif classification == 7:  # flat
            flat_note = True
        elif classification == 11:  # sharp / natural
            # TODO classify natural men sharp
            sharp_note = True
        elif classification == 4:
            if clef == True:  # A new line to write into
                # Todo write in a second line
                print(" Second Line")
            else:
                clef = True
        else:
            # Find whether it is up or down
            tb, top_image, bot_image = classifyNotePositionInSegment(notesImages[i])

            # Find the number of notes:
            if classification == 0 or classification == 3:
                print(" triple eighth or quadruple sixteenth")
                # Run row histogram peak finding on the stemless half
                if tb == 0:
                    # Remove Stems
                    # s, t = get_references(image)
                    top_image = binary_opening(top_image, np.ones((1, t + 4)))
                    show_images([top_image], ["after removal of stems"])
                    ##### Find Row Histogram
                    col_histogram = np.array([sum(top_image[:, i]) for i in range(top_image.shape[1])])
                    print(col_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = top_image.shape[0] // 3 - 1
                    peaks, _ = find_peaks(col_histogram, height=note_threshold)
                    number_of_peaks = len(peaks)

                    if number_of_peaks > 4:
                        number_of_peaks = number_of_peaks // 2

                    ##### Plot Peaks on histogram
                    print("peaks", peaks)
                    plt.plot(col_histogram)
                    plt.plot(peaks, col_histogram[peaks], "x")
                    plt.plot(np.zeros_like(col_histogram), "--", color="gray")
                    plt.show()
                elif tb == 1:
                    # Remove Stems

                    # s, t = get_references(image)
                    top_image = binary_opening(bot_image, np.ones((1, t + 4)))
                    show_images([bot_image], ["after removal of stems"])
                    ##### Find Row Histogram
                    col_histogram = np.array([sum(bot_image[:, i]) for i in range(bot_image.shape[1])])
                    print(col_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = bot_image.shape[0] // 3 - 1
                    peaks, _ = find_peaks(col_histogram, height=note_threshold)
                    number_of_peaks = len(peaks)

                    if number_of_peaks > 4:
                        number_of_peaks = number_of_peaks // 2

                    ##### Plot Peaks on histogram
                    print("peaks", peaks)
                    plt.plot(col_histogram)
                    plt.plot(peaks, col_histogram[peaks], "x")
                    plt.plot(np.zeros_like(col_histogram), "--", color="gray")
                    plt.show()

            elif classification == 15:  # Chord
                # TODO ADJUST CHORDS
                if tb == 0:
                    ##### Find Row Histogram
                    row_histogram = np.array([sum(top_image[i, :]) for i in range(top_image.shape[0])])
                    print(row_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = top_image.shape[1] // 3 - 1
                    peaks, _ = find_peaks(row_histogram, height=note_threshold)

                    ##### Plot Peaks on histogram
                    print("peaks", peaks)
                    plt.plot(row_histogram)
                    plt.plot(peaks, row_histogram[peaks], "x")
                    plt.plot(np.zeros_like(row_histogram), "--", color="gray")
                    plt.show()

                    ##### Find the local Minimas between the number of notes
                    # stacced_flag = 0
                    numberOfPeaks = len(peaks)

                    if numberOfPeaks == 1:
                        print("One Note i.e no Chord")
                    elif numberOfPeaks == 2:
                        print("Two Notes Chord")
                        '''
                        # #### Find the distance between tail peak and note peak to differentiate between // IN CASE 
                        # OF FILLING HOLES 
                        if peaks[0] - peaks[1] > peaks[1] // 2 or peaks[1] - peaks[0] > peaks[1] // 2:
                            print("False Identification of tail as a note")
                            numberOfPeaks -= 1
                            if max(row_histogram[peaks[0]], row_histogram[peaks[1]]) == row_histogram[peaks[0]]:
                                peaks = [peaks[0]]
                            else:
                                peaks = [peaks[1]]
                            print("peaks", peaks)
                        else:
                        '''
                        # for detecting stacced notes, we will put a threshold on the peaks and increment them
                        # accordingly print(row_histogram[peaks])
                        if row_histogram[peaks[0]] > row_histogram[peaks[1]] + row_histogram[peaks[1]] // 2 or \
                                row_histogram[peaks[0]] + row_histogram[peaks[0]] // 2 < row_histogram[peaks[1]]:
                            print("Three notes Chord Stacced!")
                            # Stacced flag for number of peaks increment
                            # stacced_flag = 1
                            numberOfPeaks += 1
                            # peaks = [np.min(peaks), np.max(peaks//2), np.max(peaks//2)]
                            # peaks.append(np.max(peaks//2))
                        # else:
                        #    localMinimas.append((peaks[0] + peaks[1]) // 2)
                    elif numberOfPeaks == 3:
                        print("Three Notes Chord")
                elif tb == 1:
                    ##### Find Row Histogram
                    row_histogram = np.array([sum(bot_image[i, :]) for i in range(bot_image.shape[0])])
                    print(row_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = bot_image.shape[1] // 3 - 1
                    peaks, _ = find_peaks(row_histogram, height=note_threshold)

                    ##### Plot Peaks on histogram
                    print("peaks", peaks)
                    plt.plot(row_histogram)
                    plt.plot(peaks, row_histogram[peaks], "x")
                    plt.plot(np.zeros_like(row_histogram), "--", color="gray")
                    plt.show()

                    ##### Find the local Minimas between the number of notes
                    # stacced_flag = 0
                    numberOfPeaks = len(peaks)

                    if numberOfPeaks == 1:
                        print("One Note i.e no Chord")
                    elif numberOfPeaks == 2:
                        print("Two Notes Chord")
                        '''
                        # #### Find the distance between tail peak and note peak to differentiate between // IN CASE 
                        # OF FILLING HOLES 
                        if peaks[0] - peaks[1] > peaks[1] // 2 or peaks[1] - peaks[0] > peaks[1] // 2:
                            print("False Identification of tail as a note")
                            numberOfPeaks -= 1
                            if max(row_histogram[peaks[0]], row_histogram[peaks[1]]) == row_histogram[peaks[0]]:
                                peaks = [peaks[0]]
                            else:
                                peaks = [peaks[1]]
                            print("peaks", peaks)
                        else:
                        '''
                        # for detecting stacced notes, we will put a threshold on the peaks and increment them
                        # accordingly print(row_histogram[peaks])
                        if row_histogram[peaks[0]] > row_histogram[peaks[1]] + row_histogram[peaks[1]] // 2 or \
                                row_histogram[peaks[0]] + row_histogram[peaks[0]] // 2 < row_histogram[peaks[1]]:
                            print("Three notes Chord Stacced!")
                            numberOfPeaks += 1
                    elif numberOfPeaks == 3:
                        print("Three Notes Chord")
                print("Number of notes in chord = ", numberOfPeaks)
            # Todo find Pitch
            indices = getfirst_staff_line(image)
            get_position(notesImages[i][1, :], indices[0, 0], t, s, tb)


