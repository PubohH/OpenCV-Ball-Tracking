import cv2
import numpy as np
from collections import deque

# Initialize the webcam
videoname = 'skid' #input video clip name
cap = cv2.VideoCapture(videoname + '.mp4') #可以直接输入视频名称，或者输入args.image后用cmd运行



# lower_color = np.array([40,130,0])
# upper_color = np.array([150,200,10])
#
#
sens = 0 #change this to adjust lower/upper hsv color tolerance
lower_color= np.array([0, 1, 75]) - sens
upper_color = np.array([123, 35, 255]) + sens


# Define a buffer to store the past 2 seconds of trajectory points
trajectory = deque(maxlen=10000)



# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.

result = cv2.VideoWriter(videoname + ' minEnclosingCircle.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, size)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the color pixels
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)
    # Convert the mask to a binary image
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Find the contours of the color objects in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        # Find the center and radius of the minimal circle that encloses the contour
        (x, y), radius = cv2.minEnclosingCircle(max_contour)

        # Only consider the circle if it has a radius greater than 20 pixels
        if radius > 20 and radius < 350 :
            # Draw the center of the circle
            center = (int(x), int(y))
            cv2.circle(frame, center, 3, (255, 0, 0), -8)
            # # Draw the boundary of the detected contour
            # cv2.drawContours(frame, [max_contour], -1, (255, 255, 255), 2)
            # # Draw the minimum enclosing circle
            # cv2.circle(frame, center, int(radius)+4, (0, 0, 255), 2)


            # Calculate the coordinates of the bounding rectangle
            x_rect = int(center[0] - radius)
            y_rect = int(center[1] - radius)
            w_rect = int(2 * radius) + 2
            h_rect = int(2 * radius) + 2

            # Draw the bounding rectangle around the minimal enclosing circle
            # cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)

            # Draw the trajectory of the circle for the past 2 seconds
            trajectory.appendleft(center)
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                cv2.line(frame, trajectory[i - 1], trajectory[i], (200, 255, 255), 3)

    # Show the frame
    result.write(frame)

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 将视频缩放，高=fy，宽=fx
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
# cap.release()
cv2.destroyAllWindows()
