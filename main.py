import cv2
import os
import datetime

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the video capture
video_capture = cv2.VideoCapture(0)

# Create the output directory if it doesn't already exist
output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the file counter
file_num = 0

while True:
    # Capture a frame from the video stream
    ret, frame = video_capture.read()

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Save the frame as a JPEG image if there are any detected faces
    if len(faces) > 0:

        # Calculate the center of the image
        image_center = (int(frame.shape[1]/2), int(frame.shape[0]/2))
        img_h, img_w, _ = frame.shape

        cv2.line(frame, (int(image_center[0]), 0), (int(image_center[0]), img_h), (255, 255, 255), 1)
        cv2.line(frame, (0, int(image_center[1])), (img_w, int(image_center[1])), (255, 255, 255), 1)

        # Draw a rectangle around each detected face and calculate the difference between the center of the detection area and the center of the image
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detection_center = (int(x + w/2), int(y + h/2))
            difference = (detection_center[0] - image_center[0], detection_center[1] - image_center[1])

            # Add text to the bottom of the image with FPS, datetime, and coordinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            fps = round(video_capture.get(cv2.CAP_PROP_FPS))
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #text = "".format()
            cv2.putText(frame, str(difference), (x, y-10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "{}x{}  |  FPS: {}  |  Date and Time: {}".format(img_w, img_h, fps, dt), (10, frame.shape[0]-10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the frame as a JPEG image
        filename = output_dir + "frame_{}.jpg".format(file_num)
        cv2.imwrite(filename, frame)
        file_num += 1

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
