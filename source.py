# import imutils as imutils
from cv2 import CV_32F

from util import *
from scipy.signal import find_peaks

# from util import *

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

staves = []
imagesNotes = split_images(rotated_copy, 1)
print(len(imagesNotes))
print([imagesNotes[0].shape[0]])
for i in range(len(imagesNotes)):
    if imagesNotes[i].shape[0] > 200:
        staves.append(imagesNotes[i])
if len(staves) == 0:
    print("ba7")
    staves = [rotated_copy]
show_images(staves)
imagesNotes = staves

shapes = ['triple_eighth_down', 'double_eighth_down', 'double_sixteenth_down', 'quadruple_sixteenth_down', "Clef",
          "double_flats", "double_sharps", "flat", "half_note", "quarter_note", "single_eighth note",
          "sharp", "whole_note", "single_sixteenth_note", "single_32th_note", "chord", "bar_line", "44", "42",
          "natural", "dots"]
# , "triple_sixteenth", "bar_line",   "single_quarter_note_head up"]
#
# x_train, y_train = training_data(shapes)
# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
# print(x_train.shape)
# print("y_train:", y_train)
# number_of_features = 13
# training_features = np.zeros((x_train.shape[0], number_of_features))
#
# for i in range(training_features.shape[0]):
#     components = find_regionprop(x_train[i])
#     print(y_train[i])
#     features = extract_features(components)
#     # print(features)
#     training_features[i, :] = features
#
# test_images = sorted(glob.glob('KNN Attempt/test/*'))
# ntest = len(test_images)
#
# true_values = [1, 1, 2, 3, 0, 4, 5, 6, 7, 8, 8, 8, 9, 10, 11, 19, 12, 13, 14, 15, 16, 17, 18, 20]
#
# knns = predict(test_images, shapes, true_values, training_features, y_train)
#
# accuracy_knn = calc_accuracy(knns, true_values, ntest)

# data = training_features
#
# # opening the csv file in 'w+' mode
# file = open('training_features_x.csv', 'w+', newline='')
#
# # writing the data into the file
# with file:
#     write = csv.writer(file)
#     write.writerows(data)
#
# data1 = y_train
# print(y_train)
# file = open('training_features_y.csv', 'w+', newline='')
#
# # writing the data into the file
# with file:
#     write = csv.writer(file)
#     write.writerow(data1)

# show_images([rotated], ["Binary"])
# imagesNotes = split_images(rotated_copy, 1)

training_features = np.asarray(pd.read_csv('training_features_x.csv', sep=',', header=None))
y_train = np.asarray(pd.read_csv('training_features_y.csv',sep=',', header=None))[0]



clef = False

f = open("demofile2.txt", "w")
if len(imagesNotes) != 1:
    f.write("{\n ")
else:
    f.write("[ ")
f.close()
print(training_features)
for image in imagesNotes:
    # image = rotated
    show_images([image], ["VIGOOO"])
    # withoutLines, staff_lines_beginnings = removeHLines(rotated_copy)
    # print("staff linessss: ", staff_lines_beginnings)

    #
    # # Remove Staff TIFA
    staff_indices = find_stafflines(image, 0, 0)
    print("staff linessss: ", staff_indices)
    staffbegginings, countofstaffs = countStaffLines(staff_indices)
    print("Staff beginnings:", staffbegginings)
    # rotated[staff_indices, :] = 0

    ## space = s, wel height = t
    s, t = get_references(image)
    withoutLines = binary_opening(image, np.ones((t + 2, 1)))
    # Added another opening for noise removal:
    # withoutLines = binary_opening(withoutLines, np.ones((3, 3)))
    show_images([image, withoutLines], ["Rotated", "After Line Removal"])
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
    # show_images(notesImages)

    # boxes = RetrieveComponentBox(notesImages)
    binary_notes_with_lines = segmentBoxesInImage(boxes, rotated)
    gray_notes_with_lines = segmentBoxesInImage(boxes, gray)
    # show_images(binary_notes_with_lines)

    rhythm = ["/8", "/8", "/16", "/16", "clef", "&&", "##", "&", "/2", "/4", "/8", "#", "/1", "/16", "/32", "", "",
              "", "", "", ""]
    double_flats = False
    double_sharps = False
    flat_note = False
    sharp_note = False
    natural_note = False
    four_four = False
    four_two = False
    dots = False
    olobna = []
    for i in range(len(notesImages)):
        show_images([notesImages[i]])
        io.imsave(path + str(i) + '.jpg', 1 - notesImages[i])
        test_point = extract_features_single_img(notesImages[i], areas_over_bbox[i])
        # 'triple_eighth_down', 'double_eighth_down', 'double_sixteenth_down', 'quadruple_sixteenth_down', "Clef",
        #       5    "double_flats", "double_sharps", "flat", "half_note", "quarter_note", "single_eighth note",
        #           "sharp", "whole_note", "single_sixteenth_note", "single_32th_note", "chord"
        classification = KNN(test_point, training_features, y_train, 3)
        print("Classification: ", shapes[classification], classification)
        # note_rhythm = rhythm[classification]
        # "double_flats", "double_sharps", "flat", "half_note", "quarter_note", "single_eighth note",
        #           "sharp", "whole_note", "single_sixteenth_note", "single_32th_note", "chord"

        if classification == 16:
            print("bar line")
        elif classification == 5:  # double flats
            double_flats = True
        elif classification == 6:  # double sharps
            double_sharps = True
        elif classification == 7:  # flat
            flat_note = True
        elif classification == 11:  # sharp / natural
            sharp_note = True
        elif classification == 19:
            natural_note = True
        elif classification == 20:
            dots = True
        elif classification == 4:
            if clef == True:  # A new line to write into
                f = open("demofile2.txt", "w")
                f.write(" ]\n[ ")
                f.close()
            elif clef == False and len(imagesNotes) != 1:
                clef = True
                f = open("demofile2.txt", "w")
                f.write("[ ")
                f.close()
                print(" Second Line")
            elif clef == False and len(imagesNotes) == 1:
                clef = True

        elif classification == 17:  # sharp / natural
            print("4/4")
            four_four = True
        elif classification == 18:  # sharp / natural
            print("4/2")
            four_two = True
        else:
            # Find whether it is up or down
            uu_copy = notesImages[i].copy()
            stem_removed_for_head_tail = binary_opening(uu_copy, np.ones((1, t + 4)))
            tb, _, _ = classifyNotePositionInSegment(stem_removed_for_head_tail)
            top_image = notesImages[i][0:notesImages[i].shape[0] // 2, :]
            bot_image = notesImages[i][notesImages[i].shape[0] // 2: notesImages[i].shape[0] - 1, :]
            peaks = []
            numberOfPeaks = 0
            # TODO Find the number of notes in beamed and their pitch:
            if classification == 1:
                # Run row histogram peak finding on the stemless half
                if tb == 0:
                    top_image = binary_opening(top_image, np.ones((1, t + 4)))
                    show_images([top_image], ["after removal of stems"])
                    ##### Find Row Histogram
                    col_histogram = np.array([sum(top_image[:, i]) for i in range(top_image.shape[1])])
                    print(col_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = top_image.shape[0] // 3 - 1
                    peaks, _ = find_peaks(col_histogram, height=note_threshold)
                    number_of_peaks_beamed = len(peaks)

                    if number_of_peaks_beamed > 4:
                        number_of_peaks_beamed = number_of_peaks_beamed // 2

                    ##### Plot Peaks on histogram
                    print("peaks", peaks)
                    plt.plot(col_histogram)
                    plt.plot(peaks, col_histogram[peaks], "x")
                    plt.plot(np.zeros_like(col_histogram), "--", color="gray")
                    plt.show()
                elif tb == 1:
                    # Remove Stems
                    bot_image = binary_opening(bot_image, np.ones((1, t + 4)))
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

            elif classification == 0 or classification == 3:  # or classification == 1:
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
                    number_of_peaks_beamed = len(peaks)

                    if number_of_peaks_beamed > 4:
                        number_of_peaks_beamed = number_of_peaks_beamed // 2

                    ##### Plot Peaks on histogram
                    print("peaks", peaks)
                    plt.plot(col_histogram)
                    plt.plot(peaks, col_histogram[peaks], "x")
                    plt.plot(np.zeros_like(col_histogram), "--", color="gray")
                    plt.show()
                elif tb == 1:
                    # Remove Stems

                    # s, t = get_references(image)
                    bot_image = binary_opening(bot_image, np.ones((1, t + 4)))
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
                if tb == 0:
                    # s, t = get_references(image)
                    top_image = binary_opening(top_image, np.ones((1, t + 4)))
                    show_images([top_image], ["after removal of stems"])
                    ##### Find Row Histogram
                    row_histogram = np.array([sum(top_image[i, :]) for i in range(top_image.shape[0])])
                    print(row_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = 3 * top_image.shape[1] // 4 - 1
                    peaks, _ = find_peaks(row_histogram, height=note_threshold)
                    new_peaks = []
                    for p in range(len(peaks)):
                        if p != 0:
                            if peaks[p] > peaks[p - 1] + 10:
                                new_peaks.append(peaks[p])
                        else:
                            new_peaks.append(peaks[p])
                    peaks = new_peaks
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
                    # s, t = get_references(image)
                    # bot_image = binary_opening(bot_image, np.ones((1, t + 4)))
                    show_images([bot_image], ["after removal of stems"])
                    ##### Find Row Histogram
                    row_histogram = np.array([sum(bot_image[i, :]) for i in range(bot_image.shape[0])])
                    print(row_histogram.shape)

                    ##### IF chord or not, i.e: Find Peaks corresponding to each note with the threshold
                    note_threshold = 3 * bot_image.shape[1] // 4 - 1
                    peaks, _ = find_peaks(row_histogram, height=note_threshold)
                    new_peaks = []
                    for p in range(len(peaks)):
                        if p != 0:
                            if peaks[p] > peaks[p - 1] + 10:
                                new_peaks.append(peaks[p])
                        else:
                            new_peaks.append(peaks[p])
                    peaks = new_peaks
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
                    for o in range(numberOfPeaks):
                        peaks[o] += top_image.shape[0]
                print("Number of notes in chord = ", numberOfPeaks)
                note_str = ""
                if classification == 15:  # a chord
                    # pitch = '{'
                    pitch = ""
                    octaver = ""
                    print("peaks", peaks)
                    minR, minC, maxR, maxC = boxes[i]
                    for k in range(numberOfPeaks):
                        petsh, octave = beamed_note_pitch_octave(minR + peaks[k], staffbegginings[0], t, s)
                        if petsh is not None:
                            if k != numberOfPeaks - 1:
                                if k == 0:
                                    pitch = petsh
                                    octaver = octave
                                else:
                                    pitch += petsh
                                    octaver += octave
                            else:
                                pitch += petsh
                                octaver += octave
                        print("petsh = ", petsh )
                    # pitch += '}'
                    pitch = sorted(pitch)
                    octaver = sorted(octaver)
                    output_pitch = "{"
                    for l in range(len(pitch)):
                        if l != numberOfPeaks - 1:
                            output_pitch += pitch[l] + octaver[l] + "/4 "
                        else:
                            output_pitch += pitch[l] + octaver[l] + "/4}"
                    pitch = None
                    pitch = output_pitch
                    print("pitch =", pitch)
            elif classification == 12: # whole note
                minR, minC, maxR, maxC = boxes[i]
                pitch, octave = beamed_note_pitch(minR + notesImages[i].shape[0], staffbegginings[0], t, s)
            else:
                print(staffbegginings[0])
                print("tb:", tb)
                pitch, octave = single_note_pitch(boxes[i], staffbegginings[0], t, s, tb)
            if classification != 15:
                if rhythm[classification] != "" and pitch is not None:
                    accidental = None
                    if double_sharps:
                        double_sharps = False
                        accidental = "##"
                    elif double_flats:
                        double_flats = False
                        accidental = "&&"
                    elif flat_note:
                        flat_note = False
                        accidental = "&"
                    elif sharp_note:
                        sharp_note = False
                        accidental = "#"
                    elif natural_note:
                        natural_note = False
                    if accidental is None:
                        note_str = pitch + octave + rhythm[classification]
                    else:
                        note_str = pitch + accidental + octave + rhythm[classification]
            elif rhythm[classification] == "" and pitch is not None and classification == 15:
                note_str = pitch
            if note_str != "":
                olobna.append(note_str)
    # Todo write in file
    meter = ""
    if four_two:
        print("Meter is 4/2")
        meter = '\meter<"4/2">'
        # f = open("demofile2.txt", "a")
        # f.write(meter)
        # f.close()
    elif four_two == False and four_four == True:
        print("Mtere is 4/4")
        meter = '\meter<"4/4">'
        # f = open("demofile2.txt", "a")
        # f.write(meter)
        # f.close()
    elif four_four == False and four_two == False:
        print("No meter")

    if meter != "":
        output_line = meter + " "
    else:
        output_line = None
    for h in range(len(olobna)):
        if h == 0 and output_line is None:
            output_line = olobna[h] + " "
        elif h != len(olobna) - 1:
            output_line += olobna[h] + " "
        else:
            output_line += olobna[h]
    print("output:", output_line)
    f = open("demofile2.txt", "a")
    f.write(output_line)
    f.close()

f = open("demofile2.txt", "a")
if len(imagesNotes) == 1:
    f.write(" ]")
else:
    f.write(" ]\n} ")
f.close()
f.close()
