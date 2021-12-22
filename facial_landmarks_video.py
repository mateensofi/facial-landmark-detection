# Import the necessary packages
from imutils import face_utils
import argparse
import imutils
import cv2
import dlib


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

# Initialize dlib's face detector (HOG-based) then
# create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

video = cv2.VideoCapture(0)

while True:
    # Capture video frame by frame
    ret, frame = video.read()

    # Load the frame, resize it and convert to grayscale
    image = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)

    # Detect the face in the grayscale image
    rects = detector(gray, 1)

    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        # Determine the facial landmarks for the face region then
        # convert (x, y)-coordinates to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Convert dlib rect to OpenCV style bounding-box then
        # draw the face bounding-box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the face number
        cv2.putText(image, "Face #{}".format(i+1), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Loop over the (x, y)-coordinates for the facial landmarks
        # then draw them on image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # Show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video object after the loop
video.release()

# Destroy all windows
cv2.destroyAllWindows()
