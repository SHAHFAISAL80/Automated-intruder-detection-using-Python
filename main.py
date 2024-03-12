import cv2
import winsound

# Set the video file path
video_path = r"E:\best project 9\cctv-main\theifs.mp4"
alarm_sound_path = r'E:\best project 9\cctv-main\alert.wav'

# Open the video capture object
cam = cv2.VideoCapture(video_path)

# Check if the video capture object is opened successfully
if not cam.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop to process video frames
while True:
    # Read two consecutive frames
    ret, frame1 = cam.read()

    # Check if the frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame.")
        break

    ret, frame2 = cam.read()

    # Check if the frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Calculate the absolute difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert the difference to grayscale
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Threshold the blurred image to create a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the binary image to enhance the detected regions
    dilate = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through detected contours
    for c in contours:
        # Skip contours with small area
        if cv2.contourArea(c) < 2000:
            continue

        # Get the bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(c)

        # Draw a rectangle around the detected object
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Play an alert sound
        winsound.PlaySound(alarm_sound_path, winsound.SND_ASYNC)

        # Wait for a short time
        cv2.waitKey(10)

    # Check for user input to exit the loop
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('p'):
        # Pause the video capture when 'p' is pressed
        cv2.waitKey(0)

    # Display the frame with drawn rectangles
    cv2.imshow("Camera", frame1)

# Release the video capture object and close all windows
cam.release()
cv2.destroyAllWindows()
