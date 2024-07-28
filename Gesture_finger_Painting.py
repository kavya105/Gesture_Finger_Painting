import cv2 
# CV library for computer vision tasks

import numpy as np
# Numpy for numerical operations

import mediapipe as mp
# Mediapipe for hand tracking

from collections import deque
# A double-ended queue used for hand tracking

# deque: Stands for "double-ended queue." It’s a special kind of list in Python that allows you to efficiently add or remove items from both ends. In this case, we’re using it to keep track of points on the canvas as you draw.

white_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
black_points = [deque(maxlen=1024)]

# Purpose of Color Points Arrays
# The code is designed to handle drawing on a canvas with different colors. To achieve this, it keeps track of the points (or coordinates) where the user is drawing. For each color, a separate deque is used to store these points.

# white_points: This is a list that contains one deque. The deque will store the coordinates where the user draws using the white color.
# green_points: This is a similar list that contains one deque for storing green color drawing points.
# red_points: Another list with one deque for red color drawing points.
# black_points: And another list with one deque for black color drawing points.

# Why Use deque?
# maxlen=1024: This means each deque can hold up to 1024 points. Once it reaches this limit, older points are automatically removed to make space for new ones. This helps in managing memory and ensures that only recent drawing points are stored.
# Efficiency: deque is efficient for operations like adding new points and removing old ones, which is essential for real-time drawing applications.

white_idx = 0
green_idx = 0
red_idx = 0
black_idx = 0

# Indexes to keep track of the current point list for each color.

dilation_kernel = np.ones((5,5) , np.uint8)
# dilation_kernel: This is a matrix (or array) used in image processing. In particular, it’s used for a process called dilation.

np.ones((5, 5), np.uint8)

# np.ones: This function from the NumPy library creates an array of a specified shape, where all elements are set to 1.
# (5, 5): This specifies the shape of the array. Here, it creates a 5x5 matrix.
# np.uint8: This is the data type of the array elements. np.uint8 stands for "unsigned 8-bit integer," which means each element in the array is an integer from 0 to 255.

color_palette = [(255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
current_color_index = 0
# Define colors for drawing and set the current color index to 0 (white).

canvas_width = 600
canvas_height = 471
paint_canvas = np.zeros((canvas_height , canvas_width , 3), dtype = np.uint8)+255
# + 255:

# + 255: This converts all the black pixels (value 0) to white pixels (value 255). In the color image, each pixel has three values (one for each color channel: Red, Green, Blue), and 255 means the color is fully bright in each channel.


# Draw color selection buttons on the canvas
paint_canvas = cv2.rectangle(paint_canvas, (40, 1), (140, 65), (0, 0, 0), 2)
paint_canvas = cv2.rectangle(paint_canvas, (160, 1), (255, 65), (255, 255, 255), 2)
paint_canvas = cv2.rectangle(paint_canvas, (275, 1), (370, 65), (0, 255, 0), 2)
paint_canvas = cv2.rectangle(paint_canvas, (390, 1), (485, 65), (0, 0, 255), 2)
paint_canvas = cv2.rectangle(paint_canvas, (505, 1), (600, 65), (0, 0, 0), 2)

cv2.putText(paint_canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_canvas, "WHITE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_canvas, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_canvas, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_canvas, "BLACK", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# cv2.rectangle: This function draws a rectangle on an image. The syntax is:
# cv2.rectangle(image, start_point, end_point, color, thickness)


# cv2.putText: This function adds text to an image. The syntax is:
# cv2.putText(image, text, position, font, font_scale, color, thickness, line_type) result = hands.process(frame_rgb)


# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# mp_hands = mp.solutions.hands: Creates a shortcut to the MediaPipe hand tracking solution.
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7): Initializes the hand tracking model with the ability to detect up to one hand and a confidence threshold of 70%.
# mp_draw = mp.solutions.drawing_utils: Creates a shortcut to the drawing utilities for visualizing hand landmarks and connections.

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
    # Open the webcam for capturing video. Exit if the webcam cannot be accessed.
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
     
# while True:: Starts an infinite loop to continuously process video frames.
# ret, frame = cap.read(): Reads a frame from the webcam.
# if not ret:: Checks if the frame was successfully captured.
# print("Error: Failed to capture image."): Prints an error message if the frame capture fails.
# break: Exits the loop if an error occurs.

# This loop is designed to keep processing frames from the webcam until there’s a failure in capturing the video feed, at which point it will stop processing and exit.frame_height, frame_width, _ = frame.shape

    frame_height, frame_width, _ = frame.shape
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
#     frame_height, frame_width, _ = frame.shape: Extracts the dimensions of the current frame (height and width) from the frame's shape.
# key = cv2.waitKey(1) & 0xFF: Captures the ASCII value of the key pressed (if any) during a brief wait.
# if key == ord('q'):: Checks if the pressed key is 'q'.
# break: Exits the loop if the 'q' key is pressed.
# This snippet allows the program to continuously process video frames while checking for user input. If the user presses the 'q' key, the loop will exit, and the video processing will stop.frame = cv2.flip(frame, 1)
   
   
   # Flip the frame vertically
    frame = cv2.flip(frame , 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # Resize frame to match paint_canvas width
    frame = cv2.resize(frame, (canvas_width, canvas_height))

# 0: Flips the image vertically.
# 1: Flips the image horizontally.
# -1: Flips the image both vertically and horizontally.
# frame = cv2.flip(frame, 1): This line flips the image horizontally. In the context of a webcam feed, this creates a mirror effect, which can be useful for applications where you want the image to appear as it would if you were looking into a mirror.

# cv2.cvtColor: This function converts an image from one color space to another. It takes two arguments:
# frame: The input image in its current color space.
# cv2.COLOR_BGR2RGB: The color conversion code indicating that the input image is in BGR color space and should be converted to RGB color space.
# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): This line converts the color space of the frame from BGR (Blue, Green, Red) to RGB (Red, Green, Blue).
         
         # Draw color selection buttons on the frame (for visualization)
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 255, 255), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "WHITE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
     
     
    result = hands.process(frame_rgb)
    # Process the frame to detect hand landmarks.
      
      
      
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * canvas_width)  # Scale to canvas width
                lmy = int(lm.y * canvas_height)  # Scale to canvas height
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        forefinger_pos = (landmarks[8][0], landmarks[8][1])
        finger_tip = forefinger_pos
        dot_radius = 8  # Increase the size of the dot for better visibility

        # Draw a large dot (or circle) at the finger tip
        cv2.circle(frame, finger_tip, dot_radius, (0, 255, 0), -1)  # Draw dot on frame
        cv2.circle(paint_canvas , finger_tip , dot_radius , (0 , 255 , 0 , -1))
        
        thumb_tip = (landmarks[4][0] , landmarks[4][1])
        
        
        if(thumb_tip[1]-finger_tip[1]<30):
            white_points.append(deque(maxlen=512))
            white_idx += 1
            green_points.append(deque(maxlen=512))
            green_idx += 1
            red_points.append(deque(maxlen=512))
            red_idx += 1
            black_points.append(deque(maxlen=512))
            black_idx += 1
            
#             thumb_tip[1]: The y-coordinate of the thumb tip.
# finger_tip[1]: The y-coordinate of the index finger tip.
# < 30: Checks if this vertical distance is less than 30 pixels. If true, it implies that the thumb is close to or overlapping the index finger, which can be used to trigger a specific action.

# white_points.append(deque(maxlen=512)): Starts a new drawing path for the white color if the thumb is close to the index finger. Similar actions are taken for other colors.
# This section of code is essentially checking if the user is pinching (thumb close to index finger) and, if so, starts a new drawing path for each color. This helps in organizing and managing different colors for drawing on the canvas.
           
        elif finger_tip[1]<=65: 
            if 40<=finger_tip[0]<=140:
                white_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)] 
                red_points = [deque(maxlen=512)] 
                black_points = [deque(maxlen=512)]
                
                
                white_idx = 0
                green_idx=0
                red_idx=0
                black_idx=0
                
                paint_canvas[67:,:,:] = 255 
                
#  finger_tip[1] <= 65: Checks if the y-coordinate of the finger tip is within the top 65 pixels of the frame. This area is where the color selection buttons and the clear button are located.
# if 40 <= finger_tip[0] <= 140: Checks if the x-coordinate of the finger tip is within the horizontal range of the "Clear" button (between 40 and 140 pixels).
# white_points = [deque(maxlen=512)]: Clears all drawing paths for the white color by resetting white_points to a new list with one deque.
# Similar operations are performed for green_points, red_points, and black_points.
# paint_canvas[67:, :, :] = 255: Clears the paint canvas by setting all pixels from row 67 downwards to white (255 is the color value for white).
            
            elif 160<=finger_tip[0]<=255:
                current_color_index=1  #white
            elif 275 <= finger_tip[0] <= 370:
                current_color_index = 2  # Green
            elif 390 <= finger_tip[0] <= 485:
                current_color_index = 3  # Red
            elif 505 <= finger_tip[0] <= 600:
                current_color_index = 4  # Black


# elif 160 <= finger_tip[0] <= 255: Checks if the x-coordinate of the finger tip is within the horizontal range of the "White" color button (between 160 and 255 pixels).
# current_color_index = 1: Sets the current_color_index to 1, indicating the user selected the white color.
# Similar checks and assignments are done for green, red, and black colors.
        else:
            if current_color_index == 1:
                white_points[white_idx].appendleft(finger_tip)
            elif current_color_index == 2:
                green_points[green_idx].appendleft(finger_tip)
            elif current_color_index == 3:
                red_points[red_idx].appendleft(finger_tip)
            elif current_color_index == 4:
                black_points[black_idx].appendleft(finger_tip)
    else:
        white_points.append(deque(maxlen=512))
        white_idx += 1
        green_points.append(deque(maxlen=512))
        green_idx += 1
        red_points.append(deque(maxlen=512))
        red_idx += 1
        black_points.append(deque(maxlen=512))
        black_idx += 1
        
#         else:: If the finger tip is not within the top 65 pixels (meaning it’s in the drawing area).
# if current_color_index == 1:: If the selected color is white (index 1).
# white_points[white_idx].appendleft(finger_tip): Adds the current finger tip position to the current deque for white color.
# Similar operations are done for green, red, and black colors based on the current_color_index.
            
    # Draw lines of all the colors on the canvas and frame
    point_groups = [white_points, green_points, red_points, black_points]
    for i in range(len(point_groups)):
        for j in range(len(point_groups[i])):
            for k in range(1, len(point_groups[i][j])):
                if point_groups[i][j][k - 1] is None or point_groups[i][j][k] is None:
                    continue
                # Scale points to frame size
                pt1 = (int(point_groups[i][j][k - 1][0] * (frame.shape[1] / canvas_width)), 
                       int(point_groups[i][j][k - 1][1] * (frame.shape[0] / canvas_height)))
                pt2 = (int(point_groups[i][j][k][0] * (frame.shape[1] / canvas_width)), 
                       int(point_groups[i][j][k][1] * (frame.shape[0] / canvas_height)))
                cv2.line(frame, pt1, pt2, color_palette[i], 2)
                cv2.line(paint_canvas, point_groups[i][j][k - 1], point_groups[i][j][k], color_palette[i], 2)


# point_groups: A list containing the drawing points for all colors.
# for i in range(len(point_groups)):: Iterates over each color group (white, green, red, black).
# for j in range(len(point_groups[i])):: Iterates over each deque for the current color.
# for k in range(1, len(point_groups[i][j])):: Iterates over the points in the current deque, starting from the second point.
# if point_groups[i][j][k - 1] is None or point_groups[i][j][k] is None:: Checks if either the current or previous point is None (i.e., there is no point to draw).
# pt1 and pt2: Scale the drawing points to match the frame size. This ensures that the drawing points align with the resized frame.
# cv2.line(frame, pt1, pt2, color_palette[i], 2): Draws a line on the frame from pt1 to pt2 using the color from color_palette[i].
# cv2.line(paint_canvas, point_groups[i][j][k - 1], point_groups[i][j][k], color_palette[i], 2): Draws the same line on the paint canvas.
   
   
   
    # Show frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Paint', paint_canvas)
    
#     cv2.imshow('Frame', frame): Displays the webcam feed with the detected hand landmarks and drawing.
# cv2.imshow('Paint', paint_canvas): Displays the paint canvas with the drawings made using the hand gestures.

cap.release()
cv2.destroyAllWindows()
            
#             cap.release(): Releases the webcam capture object, freeing up the camera for other applications.
# cv2.destroyAllWindows(): Closes all OpenCV windows opened during the program execution.