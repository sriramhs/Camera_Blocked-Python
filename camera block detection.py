import cv2
import numpy as np

# Set the threshold for blocked camera detection prev frame
threshold1 = 0.99999

# Set the threshold for blocked camera detection hist
threshold2 = 10000

# Open the video capture device or file
cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index or video file path

# Read the first frame from the video capture device or file
_, prev_frame = cap.read()

while True:
    # Read the current frame from the video capture device or file
    _, current_frame = cap.read()

    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(current_frame, prev_frame)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Compute the percentage of pixels that are similar between the two frames
    similarity = np.mean(gray < 10)
    
     # Compute the histogram of the grayscale image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Compute the mean and standard deviation of the histogram
    mean, std_dev = cv2.meanStdDev(hist)

    # Check if the camera is blocked
    if (similarity > threshold1) and (std_dev > threshold2):
        print('Camera is blocked')
        continue
    else:
        print('Camera is not blocked')

    # Display the current frame on the screen
    cv2.imshow('Frame', current_frame)

    # Wait for a key press and then exit if the key is the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the current frame as the previous frame for the next iteration
    prev_frame = current_frame

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
