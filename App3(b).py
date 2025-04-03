# Added More shapes: Cube, Cubiod, Triangle, Square

#importing necessary libraries
from flask import Flask, render_template, Response
import os
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import speech_recognition as sr
import time
import pyaudio
from datetime import datetime

app = Flask(__name__)
app.config['selected_mic'] = None
app.config['mics'] = None
# Voice Command Listener Class
class VoiceCommandListener:
    def __init__(self, device_index=None):
        self.recognizer = sr.Recognizer()
        self.device_index = device_index
        self.microphone = sr.Microphone(device_index=device_index)
        self.command_queue = queue.Queue()
        self.running = False
        self.listen_thread = None
        
        self.shape_commands = {
            "draw circle": "Circle",
            "draw rectangle": "Rectangle",
            "draw triangle": "Triangle",
            "draw square": "Square",
            "draw cube": "Cube",
            "draw cuboid": "Cuboid"
        }
    
    @staticmethod
    def list_microphones():
        """Static method to list all available microphones"""
        p = pyaudio.PyAudio()
        mics = []
        
        # List all audio devices
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            # Only include input devices
            if dev_info.get('maxInputChannels') > 0:
                print(f"Device {i}: {dev_info.get('name')}")
                mics.append((i, dev_info.get('name')))
        
        p.terminate()
        return mics
    
    def set_microphone(self, device_index):
        """Change the microphone being used"""
        if self.running:
            # Stop current thread before changing microphone
            self.stop_listening()
        
        self.device_index = device_index
        self.microphone = sr.Microphone(device_index=device_index)
        
        if self.running:
            # Restart thread with new microphone
            self.start_listening()
    
    def start_listening(self):
        """Start the voice command listener thread"""
        if not self.running:
            self.running = True
            self.listen_thread = threading.Thread(target=self._listen_loop)
            self.listen_thread.daemon = True  # Thread will exit when main program exits
            self.listen_thread.start()
            print(f"Voice command listener started using device index {self.device_index}.")
    
    def stop_listening(self):
        """Stop the voice command listener thread"""
        self.running = False
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1)
            print("Voice command listener stopped.")
    
    def _listen_loop(self):
        """Background thread function that continuously listens for commands"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Background noise level adjusted")
            
            # Recognize speech using GoogleSpeech API
            while self.running:
                try:
                    with self.microphone as source:
                        print("Listening for commands...")
                        audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
                    
                    try:
                        command = self.recognizer.recognize_google(audio).strip().lower()
                        print(f"Detected: {command}")
                        
                        # Check for valid commands
                        if "screenshot" in command:
                            self.command_queue.put("screenshot")
                        elif "clear" in command:
                            self.command_queue.put("clear")
                        elif "increase" in command:
                            self.command_queue.put("increase")
                        elif "decrease" in command: 
                            self.command_queue.put("decrease")

                        for shape_phrase, shape_tool in self.shape_commands.items():
                            if shape_phrase in command:
                                self.command_queue.put(f"draw {shape_tool}")
                                break
                            
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        print(f"Speech service error: {e}")
                
                except Exception as e:
                    print(f"Error in voice listener: {e}")
                    time.sleep(1)  # Add a delay to prevent CPU overload if there's a persistent error
        
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            self.running = False
    
    def get_command(self):
        """
        Non-blocking method to check if a command is available
        Returns the command string or None if no command is available
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

# webcam initialization

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
cap.set(3, 1280)  # (3)width
cap.set(4, 720)   # (4)height
cap.set(10, 150)  # brightness=150%

def generate_frames():
    # Canvas
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    global canvas
    canvas = np.zeros((720, 1280, 3), np.uint8)
    canvas[:, :, :] = 255
    # Black Canvas
    canvasBlack = np.zeros((720, 1280, 3), np.uint8)

    # # header bar image
    # overlay = cv2.imread("images/sam3.png")[0:80, 0:1280]
    overlay_path = os.path.join(static_folder, 'images', 'BarSide1.png')
    # # SideBar image
    overplay_img = cv2.imread(overlay_path)
    # sidebar = cv2.imread("images/sam3.png")[80:720, 1200:1280]

    if overplay_img is None:
        print(f"Error: Could not load image from {overlay_path}")
        overlay = np.zeros((80, 1280, 3), np.uint8)
    else:
        overlay = overplay_img[0:80, 0:1280]    
        
    sidebar_path = os.path.join(static_folder, 'images', 'BarSide1.png')
    sidebar_img = cv2.imread(sidebar_path)
    
    if sidebar_img is None:
        print(f"ERROR: Could not load image from {sidebar_path}")
        # Provide a fallback
        sidebar = np.zeros((640, 80, 3), np.uint8)
    else:
        sidebar = sidebar_img[80:720, 1200:1280]
    
    # Mediapipe hand object
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()  # hands=mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    # Mediapipes Drawing tool for connecting hand landmarks
    mp_draw = mp.solutions.drawing_utils

    # tools
    drawColor = (0, 0, 255)

    # for displaying on screen
    selectedColor = 'Blue'
    selectedTool = 'Draw'

    tool = "Draw"  # important for selection

    xp, yp = 0, 0  # previous position of index finger

    # variables for shapes
    countCircle = 1
    countRectangle = 1
    countTriangle = 1
    countSquare = 1
    countCube = 1
    countCuboid = 1

    # Pointer thickness control variavles

    brushThickness = 10
    eraserThickness = 30 
    minThickness = 2
    maxThickness = 40
    maxEraserThickness = 100
    thicknessStep = 3 #step increase during voice command increase method
    # minDistance = 20
    # maxDistance = 150



    # Initialize voice listener
    voice_listener = None
    if app.config['mics']:
        voice_listener = VoiceCommandListener(device_index=app.config['selected_mic'])
        voice_listener.start_listening()

    # Function for finding how much fingers are up
    tipIds = [8, 12, 16, 20]  # finger tip ids except for thump tip (4)
    def fingerUp(landmark):
        fingerList = []
        # thump up/down finding is different if thumptip(4) is left to id 3 then up else down ie, x(id4)<x(id3)
        # if landmark[4][1] < landmark[3][1]:
        #     fingerList.append(1)  # 0-id 1-x 2-y in landmark
        # else:
        #     fingerList.append(0)
        
        handType = results.multi_handedness[0].classification[0].label  # 'Right' or 'Left'

        if handType == 'Right':
            if landmark[4][1] < landmark[3][1]:  # Thumb is Up for Right Hand
                fingerList.append(1)
            else:
                fingerList.append(0)
        else:  # Left Hand
            if landmark[4][1] > landmark[3][1]:  # Thumb is Up for Left Hand (Opposite Condition)
                fingerList.append(1)
            else:
                fingerList.append(0)

        # For the rest of fingers if y(id-tip)<y(id-middlepart) then up else down (id-2 bcz middle part of finger)
        for id in tipIds:
            if landmark[id][2] < landmark[id - 2][2]:
                fingerList.append(1)
            else:
                fingerList.append(0)

        return fingerList

    # calculate distance between 2 points(fingers)
    def calculate_distance(pt1, pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5


    # For showing voice command status
    last_command_time = 0
    command_display = ""
    command_display_duration = 2  # seconds
    try:
        while cap.isOpened():
            success, img = cap.read()
            
            # Check if frame was successfully captured
            if not success:
                print("Failed to capture frame from webcam")
                continue
                
            # flipping to make correct aligned(1=horizontally)
            img = cv2.flip(img, 1)

            # 2. Landmark and position finding
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bcz mediapipe prefer RGB as it is trained in RGB
            results = hands.process(imgRGB)  # Hand Detected

            landMark = []  # Landmark list for storing position of each finger

            # Check for voice commands (non-blocking)
            if voice_listener:
                try:
                    command = voice_listener.get_command()
                    if command:
                        if command == "screenshot":
                            print("Saving the canvas...")
                            # Save the canvas to a file with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"phantom_canvas_{timestamp}.png"
                            cv2.imwrite(filename, canvas)
                            print(f"Canvas saved as {filename}")
                            command_display = f"Saved as {filename}"
                            last_command_time = time.time()
                        elif command == "clear":
                            print("Clearing the canvas...")
                            canvasBlack = np.zeros((720, 1280, 3), np.uint8)
                            canvas[:, :, :] = 255
                            command_display = "Canvas cleared"
                            last_command_time = time.time()
                        elif command == "increase":
                            if tool == "Eraser":
                                eraserThickness = min(eraserThickness + thicknessStep*2, maxEraserThickness)
                                command_display = f"Eraser thickness: {eraserThickness}"
                            else:
                                brushThickness = min(brushThickness + thicknessStep, maxThickness)
                                command_display = f"Brush thickness: {brushThickness}"
                            last_command_time=time.time()
                        elif command == "decrease":
                            if tool == "Eraser":
                                eraserThickness = max(eraserThickness - thicknessStep*2, minThickness)
                                command_display = f"Eraser thickness: {eraserThickness}"
                            else:
                                brushThickness = max(brushThickness - thicknessStep, minThickness)
                                command_display = f"Brush thickness: {brushThickness}"
                            last_command_time = time.time()
                        elif command.startsWith("draw"):
                            shape = command.split(" ")[1]
                            if results.multi_hand_landmarks:
                                xi, yi = landMark[8][1:]
                                if shape =="Circle":
                                    tool = "Circle"
                                    countCircle = 0
                                    xstart_circle, ystart_circle = xi, yi
                                    command_display = f"Drawing {shape}"
                                    last_command_time = time.time()
                                elif shape == "Rectangle":
                                    tool = "Rectangle"
                                    countRectangle = 0
                                    xstart_rect, ystart_rect = xi, yi
                                    command_display = f"Drawing {shape}"
                                    last_command_time = time.time()
                                
                                elif shape == "Square":
                                    tool = "Square"
                                    countSquare = 0
                                    xstart_square, ystart_square = xi, yi
                                    command_display = f"Drawing {shape}"
                                    last_command_time = time.time()
                                
                                elif shape == "Triangle":
                                    tool = "Triangle"
                                    countTriangle = 0
                                    xstart_triangle, ystart_triangle = xi, yi
                                    command_display = f"Drawing {shape}"
                                    last_command_time = time.time()
                                
                                elif shape == "Cube":
                                    tool = "Cube"
                                    countCube = 0
                                    xstart_cube, ystart_cube = xi, yi
                                    command_display = f"Drawing {shape}"
                                    last_command_time = time.time()
                                
                                elif shape == "Cuboid":
                                    tool = "Cuboid"
                                    countCuboid = 0
                                    xstart_cuboid, ystart_cuboid = xi, yi
                                    command_display = f"Drawing {shape}"
                                    last_command_time = time.time()
                                                
                except Exception as e:
                    print(f"Error processing voice command: {e}")
            
            # if hand is detected
            if results.multi_hand_landmarks:
                lndmrk = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(img, lndmrk, mp_hands.HAND_CONNECTIONS)  # drawing connection not necessary

                for id, lm in enumerate(lndmrk.landmark):
                    height, width, _ = img.shape
                    # this is done because lm.x gives ratio of x position but we need pixel value so multiply by width , same for height
                    x, y = int(lm.x * width), int(lm.y * height)
                    # appending each landmarks id and position as a list to landMark
                    landMark.append([id, x, y])

                xi, yi = landMark[8][1:]  # index fingers position
                xm, ym = landMark[12][1:]  # middle fingers position
                # print(xm,ym)

                if len(landMark) > 4:
                    xthumb, ythumb = landMark[4][1:]  # thumb position for thickness control
                # 3.opened fingers
                fingerList = fingerUp(landMark)

                # 4. Selection Mode
                # xstart_triangle, ystart_triangle = 0, 0
                # xmid_triangle, ymid_triangle = 0, 0
                # xlast_triangle, ylast_triangle = 0, 0
                # # For square
                # xstart_square, ystart_square = 0, 0
                # xlast_square, ylast_square = 0, 0

                # # For cube
                # xstart_cube, ystart_cube = 0, 0
                # xlast_cube, ylast_cube = 0, 0

                # # For cuboid
                # xstart_cuboid, ystart_cuboid = 0, 0
                # xlast_cuboid, ylast_cuboid = 0, 0
                
                if fingerList[1] and fingerList[2]: # 2 fingers up(Selection mode)

                    # circle finishing
                    if countCircle == 0:
                        cv2.circle(img, (xstart_circle, ystart_circle), int(((xstart_circle - xlast_circle) ** 2 + (ystart_circle - ylast_circle) ** 2) ** 0.5), drawColor, brushThickness)
                        cv2.circle(canvas, (xstart_circle, ystart_circle), int(((xstart_circle - xlast_circle) ** 2 + (ystart_circle - ylast_circle) ** 2) ** 0.5), drawColor, brushThickness)
                        cv2.circle(canvasBlack, (xstart_circle, ystart_circle), int(((xstart_circle - xlast_circle) ** 2 + (ystart_circle - ylast_circle) ** 2) ** 0.5), drawColor, brushThickness)
                        countCircle = 1

                    # rectangle finishing
                    if countRectangle == 0:
                        cv2.rectangle(img, (xstart_rect, ystart_rect), (xlast_rect, ylast_rect), drawColor, brushThickness)
                        cv2.rectangle(canvas, (xstart_rect, ystart_rect), (xlast_rect, ylast_rect), drawColor, brushThickness)
                        cv2.rectangle(canvasBlack, (xstart_rect, ystart_rect), (xlast_rect, ylast_rect), drawColor, brushThickness)
                        countRectangle = 1
                        
                    # Triangle finishing
                    if countTriangle == 0:
                    # Create a numpy array of triangle vertices
                        triangle_pts = np.array([[xstart_triangle, ystart_triangle], [xmid_triangle, ymid_triangle], [xlast_triangle, ylast_triangle]], np.int32) 
                        triangle_pts = triangle_pts.reshape((-1, 1, 2))
                        
                        # Draw the triangle on all canvases
                        cv2.polylines(img, [triangle_pts], True, drawColor, brushThickness)
                        cv2.polylines(canvas, [triangle_pts], True, drawColor, brushThickness)
                        cv2.polylines(canvasBlack, [triangle_pts], True, drawColor, brushThickness)
                        countTriangle = 1
                    # Square finishing
                    if countSquare == 0:
                        # Calculate side length based on diagonal points
                        side_length = min(abs(xlast_square - xstart_square), abs(ylast_square - ystart_square))
                        
                        # Define square points based on start point and side length
                        if xlast_square > xstart_square:
                            if ylast_square > ystart_square:
                                # Bottom-right quadrant
                                x2, y2 = xstart_square + side_length, ystart_square
                                x3, y3 = xstart_square + side_length, ystart_square + side_length
                                x4, y4 = xstart_square, ystart_square + side_length
                            else:
                                # Top-right quadrant
                                x2, y2 = xstart_square + side_length, ystart_square
                                x3, y3 = xstart_square + side_length, ystart_square - side_length
                                x4, y4 = xstart_square, ystart_square - side_length
                        else:
                            if ylast_square > ystart_square:
                                # Bottom-left quadrant
                                x2, y2 = xstart_square - side_length, ystart_square
                                x3, y3 = xstart_square - side_length, ystart_square + side_length
                                x4, y4 = xstart_square, ystart_square + side_length
                            else:
                                # Top-left quadrant
                                x2, y2 = xstart_square - side_length, ystart_square
                                x3, y3 = xstart_square - side_length, ystart_square - side_length
                                x4, y4 = xstart_square, ystart_square - side_length
                    
                    # Cube finishing
                    if countCube == 0:
                        # Calculate dimension
                        dim = min(abs(xlast_cube - xstart_cube), abs(ylast_cube - ystart_cube)) // 2
                        
                        # Front face vertices
                        front_top_left = (xstart_cube, ystart_cube)
                        front_top_right = (xstart_cube + dim, ystart_cube)
                        front_bottom_right = (xstart_cube + dim, ystart_cube + dim)
                        front_bottom_left = (xstart_cube, ystart_cube + dim)
                        
                        # Back face vertices (with perspective)
                        offset = dim // 2  # Perspective offset
                        back_top_left = (xstart_cube + offset, ystart_cube - offset)
                        back_top_right = (xstart_cube + dim + offset, ystart_cube - offset)
                        back_bottom_right = (xstart_cube + dim + offset, ystart_cube + dim - offset)
                        back_bottom_left = (xstart_cube + offset, ystart_cube + dim - offset)
                        
                        # Draw front face
                        cv2.line(img, front_top_left, front_top_right, drawColor, brushThickness)
                        cv2.line(img, front_top_right, front_bottom_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_right, front_bottom_left, drawColor, brushThickness)
                        cv2.line(img, front_bottom_left, front_top_left, drawColor, brushThickness)
                        
                        # Draw back face
                        cv2.line(img, back_top_left, back_top_right, drawColor, brushThickness)
                        cv2.line(img, back_top_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(img, back_bottom_right, back_bottom_left, drawColor, brushThickness)
                        cv2.line(img, back_bottom_left, back_top_left, drawColor, brushThickness)
                        
                        # Connect front and back faces
                        cv2.line(img, front_top_left, back_top_left, drawColor, brushThickness)
                        cv2.line(img, front_top_right, back_top_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_left, back_bottom_left, drawColor, brushThickness)
                        
                        # Do the same for canvas and canvasBlack
                        # Draw front face
                        cv2.line(canvas, front_top_left, front_top_right, drawColor, brushThickness)
                        cv2.line(canvas, front_top_right, front_bottom_right, drawColor, brushThickness)
                        cv2.line(canvas, front_bottom_right, front_bottom_left, drawColor, brushThickness)
                        cv2.line(canvas, front_bottom_left, front_top_left, drawColor, brushThickness)
                        
                        # Draw back face
                        cv2.line(canvas, back_top_left, back_top_right, drawColor, brushThickness)
                        cv2.line(canvas, back_top_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(canvas, back_bottom_right, back_bottom_left, drawColor, brushThickness)
                        cv2.line(canvas, back_bottom_left, back_top_left, drawColor, brushThickness)
                        
                        # Connect front and back faces
                        cv2.line(canvas, front_top_left, back_top_left, drawColor, brushThickness)
                        cv2.line(canvas, front_top_right, back_top_right, drawColor, brushThickness)
                        cv2.line(canvas, front_bottom_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(canvas, front_bottom_left, back_bottom_left, drawColor, brushThickness)
                        
                        # Repeat for canvasBlack
                        cv2.line(canvasBlack, front_top_left, front_top_right, drawColor, brushThickness)
                        cv2.line(canvasBlack, front_top_right, front_bottom_right, drawColor, brushThickness)
                        cv2.line(canvasBlack, front_bottom_right, front_bottom_left, drawColor, brushThickness)
                        cv2.line(canvasBlack, front_bottom_left, front_top_left, drawColor, brushThickness)
                        
                        cv2.line(canvasBlack, back_top_left, back_top_right, drawColor, brushThickness)
                        cv2.line(canvasBlack, back_top_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(canvasBlack, back_bottom_right, back_bottom_left, drawColor, brushThickness)
                        cv2.line(canvasBlack, back_bottom_left, back_top_left, drawColor, brushThickness)
                        
                        cv2.line(canvasBlack, front_top_left, back_top_left, drawColor, brushThickness)
                        cv2.line(canvasBlack, front_top_right, back_top_right, drawColor, brushThickness)
                        cv2.line(canvasBlack, front_bottom_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(canvasBlack, front_bottom_left, back_bottom_left, drawColor, brushThickness)
                        
                        countCube = 1

                    # Cuboid finishing
                    if countCuboid == 0:
                        # Calculate dimensions
                        width = abs(xlast_cuboid - xstart_cuboid)
                        height = abs(ylast_cuboid - ystart_cuboid)
                        depth = min(width, height) // 2  # Use a proportion of width/height for depth
                        
                        # Determine the direction based on user drawing
                        x_direction = 1 if xlast_cuboid > xstart_cuboid else -1
                        y_direction = 1 if ylast_cuboid > ystart_cuboid else -1
                        
                        # Front face vertices
                        front_top_left = (xstart_cuboid, ystart_cuboid)
                        front_top_right = (xstart_cuboid + width * x_direction, ystart_cuboid)
                        front_bottom_right = (xstart_cuboid + width * x_direction, ystart_cuboid + height * y_direction)
                        front_bottom_left = (xstart_cuboid, ystart_cuboid + height * y_direction)
                        
                        # Back face vertices
                        offset = depth // 2  # Perspective offset
                        back_top_left = (xstart_cuboid + offset, ystart_cuboid - offset)
                        back_top_right = (xstart_cuboid + width * x_direction + offset, ystart_cuboid - offset)
                        back_bottom_right = (xstart_cuboid + width * x_direction + offset, ystart_cuboid + height * y_direction - offset)
                        back_bottom_left = (xstart_cuboid + offset, ystart_cuboid + height * y_direction - offset)
                        
                        # Draw on all canvases
                        for canvas_type in [img, canvas, canvasBlack]:
                            # Draw front face
                            cv2.line(canvas_type, front_top_left, front_top_right, drawColor, brushThickness)
                            cv2.line(canvas_type, front_top_right, front_bottom_right, drawColor, brushThickness)
                            cv2.line(canvas_type, front_bottom_right, front_bottom_left, drawColor, brushThickness)
                            cv2.line(canvas_type, front_bottom_left, front_top_left, drawColor, brushThickness)
                            
                            # Draw back face
                            cv2.line(canvas_type, back_top_left, back_top_right, drawColor, brushThickness)
                            cv2.line(canvas_type, back_top_right, back_bottom_right, drawColor, brushThickness)
                            cv2.line(canvas_type, back_bottom_right, back_bottom_left, drawColor, brushThickness)
                            cv2.line(canvas_type, back_bottom_left, back_top_left, drawColor, brushThickness)
                            
                            # Connect front and back faces
                            cv2.line(canvas_type, front_top_left, back_top_left, drawColor, brushThickness)
                            cv2.line(canvas_type, front_top_right, back_top_right, drawColor, brushThickness)
                            cv2.line(canvas_type, front_bottom_right, back_bottom_right, drawColor, brushThickness)
                            cv2.line(canvas_type, front_bottom_left, back_bottom_left, drawColor, brushThickness)
                        
                        countCuboid = 1

                    # to make discontinuity after selection
                    xp, yp = 0, 0

                    cv2.rectangle(img, (xi - 10, yi - 15), (xm + 10, ym + 20), drawColor, -1)

                    # check if finger on header portion   check later y 125 not 80
                    if yi < 105:
                        # check if fingers are in which x position

                        # Red Color
                        if 20 < xm < 142:
                            drawColor = (0, 0, 255)
                            selectedColor = "Red"

                        # Blue Color
                        elif 160 < xm < 280:
                            drawColor = (255, 100, 0)
                            selectedColor = "Blue"

                        # Green Color
                        elif 315 < xm < 430:
                            drawColor = (0, 255, 0)
                            selectedColor = "Green"

                        # Yellow Color
                        elif 455 < xm < 580:
                            drawColor = (0, 255, 255)
                            selectedColor = "Yellow"
                            
                        # Violet Color
                        elif 600 < xm < 755:
                            drawColor = (211, 0, 148)
                            selectedColor = "Violet"
                        
                        # Pink Color:
                        elif 775 < xm < 890:
                            drawColor = (203, 192, 255)
                            selectedColor = "Pink"
                            
                        # Eraser:
                        elif 930 < xm < 1040:
                            drawColor = (0,0,0)
                            selectedColor = "none"
                            selectedTool = 'Eraser'
                            tool = "Eraser"
                        
                        # Draw
                        elif 1050 < xm < 1170:
                            tool = "Draw"
                            selectedTool = 'Draw'
                            command_display = "Draw selected"
                            last_command_time = time.time()
                        

                    # side tool selection
                    if 1180 < xm < 1280:
                        if 0 < ym < 105:
                            canvasBlack = np.zeros((720, 1280, 3), np.uint8)
                            canvas[:, :, :] = 255
                            command_display = "Canvas cleared"
                            last_command_time = time.time()
                            
                        elif 110 < ym < 185:
                            tool = "Triangle"
                            selectedTool = 'Triangle'
                            command_display = "Triangle selected"
                            last_command_time = time.time()
                            
                        elif 194 < ym < 280:
                            tool = "Circle"
                            selectedTool = 'Circle'
                            command_display = "Circle selected"
                            last_command_time = time.time()
                            
                        elif 290 < ym < 400:
                            tool = "Rectangle"
                            selectedTool = 'Rectangle'
                            command_display = "Rectangle selected"
                            last_command_time = time.time()
                            
                        elif 410 < ym < 490:
                            tool = "Square"
                            selectedTool = 'Square'
                            command_display = "Square selected"
                            last_command_time = time.time()
                            
                        elif 500 < ym < 590:
                            tool = "Cube"
                            selectedTool = 'Cube'
                            command_display = "Cube selected"
                            last_command_time = time.time()
                            
                        elif 600 < ym < 715:
                            tool = "Cuboid"
                            selectedTool = 'Cuboid'
                            command_display = "Cuboid selected"
                            last_command_time = time.time()
                                    
                                    
                # 5. Drawing Mode==================================================================================
                if fingerList[1] and fingerList[2] == 0:
                    current_thickness = brushThickness
                        
                    cv2.circle(img, (xi, yi), 15, drawColor, -1)

                    if tool == "Eraser":
                        # when frame start dont make a line from 0,0 so draw a line from xi,yi to xi,yi ie a point
                        if xp == 0 and yp == 0:
                            xp, yp = xi, yi
                        
                        cv2.line(img, (xp, yp), (xi, yi), drawColor, eraserThickness)
                        cv2.line(canvas, (xp, yp), (xi, yi), (255, 255, 255), eraserThickness)
                        cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, eraserThickness)

                        xp, yp = xi, yi

                

                    # Drawing
                    if tool == "Draw":
                        # when frame start dont make a line from 0,0 so draw a line from xi,yi to xi,yi ie a point
                        if xp == 0 and yp == 0:
                            xp, yp = xi, yi

                        # it is to automatically make eraser back to normal size
                        if drawColor != (0, 0, 0):
                            cv2.line(img, (xp, yp), (xi, yi), drawColor, brushThickness)
                            cv2.line(canvas, (xp, yp), (xi, yi), drawColor, brushThickness)
                            cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, brushThickness)
                        else:
                            cv2.line(img, (xp, yp), (xi, yi), drawColor, eraserThickness)
                            cv2.line(canvas, (xp, yp), (xi, yi), (255, 255, 255), eraserThickness)
                            cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, eraserThickness)
                        # update xp and yp
                        xp, yp = xi, yi
                        
                    # Circle
                    elif tool == "Circle":
                        if countCircle == 1:
                            xstart_circle, ystart_circle = xi, yi
                            countCircle = 0
                        cv2.circle(img, (xstart_circle, ystart_circle), int(((xstart_circle - xi) ** 2 + (ystart_circle - yi) ** 2) ** 0.5), drawColor, brushThickness)
                        xlast_circle, ylast_circle = xi, yi
                    # Rectanlge 
                    elif tool == "Rectangle":
                        if countRectangle == 1:
                            xstart_rect, ystart_rect = xi, yi
                            countRectangle = 0
                        cv2.rectangle(img, (xstart_rect, ystart_rect), (xi, yi), drawColor, brushThickness)
                        xlast_rect, ylast_rect = xi, yi
                        
                    # For Triangle
                    elif tool == "Triangle":
                        if countTriangle == 1:
                            xstart_triangle, ystart_triangle = xi, yi
                            countTriangle = 0
                        # Calculate third point based on current position
                        height = int(((xstart_triangle - xi)**2 + (ystart_triangle - yi)**2)**0.5)
                        angle = np.arctan2(yi - ystart_triangle, xi - xstart_triangle)
                        xthird = int(xstart_triangle + height * np.cos(angle + np.pi/2))
                        ythird = int(ystart_triangle + height * np.sin(angle + np.pi/2))
                        # Update last positions
                        xmid_triangle, ymid_triangle = xthird, ythird
                        xlast_triangle, ylast_triangle = xi, yi

                    # For Square
                    elif tool == "Square":
                        if countSquare == 1:
                            xstart_square, ystart_square = xi, yi
                            countSquare = 0
                        
                        # Calculate side length based on diagonal distance
                        side_length = max(abs(xi - xstart_square), abs(yi - ystart_square))
                        
                        # Determine direction
                        x_direction = 1 if xi > xstart_square else -1
                        y_direction = 1 if yi > ystart_square else -1
                        
                        # Calculate square corners
                        x2 = xstart_square + side_length * x_direction
                        y2 = ystart_square
                        x3 = xstart_square + side_length * x_direction
                        y3 = ystart_square + side_length * y_direction
                        x4 = xstart_square
                        y4 = ystart_square + side_length * y_direction
                        
                        # Create square points
                        square_pts = np.array([[xstart_square, ystart_square], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        square_pts = square_pts.reshape((-1, 1, 2))
                        
                        # Draw preview
                        cv2.polylines(img, [square_pts], True, drawColor, brushThickness)
                        xlast_square, ylast_square = xi, yi

                    # For Cube
                    elif tool == "Cube":
                        if countCube == 1:
                            xstart_cube, ystart_cube = xi, yi
                            countCube = 0
                        
                        # Calculate dimension based on distance
                        dim = abs(xi - xstart_cube)
                        
                        # Front face vertices
                        front_top_left = (xstart_cube, ystart_cube)
                        front_top_right = (xstart_cube + dim, ystart_cube)
                        front_bottom_right = (xstart_cube + dim, ystart_cube + dim)
                        front_bottom_left = (xstart_cube, ystart_cube + dim)
                        
                        # Back face vertices (with perspective)
                        offset = dim // 3  # Perspective offset
                        back_top_left = (xstart_cube + offset, ystart_cube - offset)
                        back_top_right = (xstart_cube + dim + offset, ystart_cube - offset)
                        back_bottom_right = (xstart_cube + dim + offset, ystart_cube + dim - offset)
                        back_bottom_left = (xstart_cube + offset, ystart_cube + dim - offset)
                        
                        # Draw front face
                        cv2.line(img, front_top_left, front_top_right, drawColor, brushThickness)
                        cv2.line(img, front_top_right, front_bottom_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_right, front_bottom_left, drawColor, brushThickness)
                        cv2.line(img, front_bottom_left, front_top_left, drawColor, brushThickness)
                        
                        # Draw back face
                        cv2.line(img, back_top_left, back_top_right, drawColor, brushThickness)
                        cv2.line(img, back_top_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(img, back_bottom_right, back_bottom_left, drawColor, brushThickness)
                        cv2.line(img, back_bottom_left, back_top_left, drawColor, brushThickness)
                        
                        # Connect front and back faces
                        cv2.line(img, front_top_left, back_top_left, drawColor, brushThickness)
                        cv2.line(img, front_top_right, back_top_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_left, back_bottom_left, drawColor, brushThickness)
                        
                        xlast_cube, ylast_cube = xi, yi

                    # For Cuboid
                    elif tool == "Cuboid":
                        if countCuboid == 1:
                            xstart_cuboid, ystart_cuboid = xi, yi
                            countCuboid = 0
                        
                        # Calculate dimensions
                        width = abs(xi - xstart_cuboid)
                        height = abs(yi - ystart_cuboid)
                        depth = min(width, height) // 2  # Depth is proportional to smaller dimension
                        
                        # Determine direction
                        x_direction = 1 if xi > xstart_cuboid else -1
                        y_direction = 1 if yi > ystart_cuboid else -1
                        
                        # Front face vertices
                        front_top_left = (xstart_cuboid, ystart_cuboid)
                        front_top_right = (xstart_cuboid + width * x_direction, ystart_cuboid)
                        front_bottom_right = (xstart_cuboid + width * x_direction, ystart_cuboid + height * y_direction)
                        front_bottom_left = (xstart_cuboid, ystart_cuboid + height * y_direction)
                        
                        # Back face vertices (with perspective)
                        offset = depth // 2  # Perspective offset
                        back_top_left = (xstart_cuboid + offset, ystart_cuboid - offset)
                        back_top_right = (xstart_cuboid + width * x_direction + offset, ystart_cuboid - offset)
                        back_bottom_right = (xstart_cuboid + width * x_direction + offset, ystart_cuboid + height * y_direction - offset)
                        back_bottom_left = (xstart_cuboid + offset, ystart_cuboid + height * y_direction - offset)
                        
                        # Draw front face
                        cv2.line(img, front_top_left, front_top_right, drawColor, brushThickness)
                        cv2.line(img, front_top_right, front_bottom_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_right, front_bottom_left, drawColor, brushThickness)
                        cv2.line(img, front_bottom_left, front_top_left, drawColor, brushThickness)
                        
                        # Draw back face
                        cv2.line(img, back_top_left, back_top_right, drawColor, brushThickness)
                        cv2.line(img, back_top_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(img, back_bottom_right, back_bottom_left, drawColor, brushThickness)
                        cv2.line(img, back_bottom_left, back_top_left, drawColor, brushThickness)
                        
                        # Connect front and back faces
                        cv2.line(img, front_top_left, back_top_left, drawColor, brushThickness)
                        cv2.line(img, front_top_right, back_top_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_right, back_bottom_right, drawColor, brushThickness)
                        cv2.line(img, front_bottom_left, back_bottom_left, drawColor, brushThickness)
                        
                        xlast_cuboid, ylast_cuboid = xi, yi
                    

            # 6 . Adding canvas and real fram
            imgGray = cv2.cvtColor(canvasBlack, cv2.COLOR_BGR2GRAY)
            _, imgBin = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgBin = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgBin)
            img = cv2.bitwise_or(img, canvasBlack)

            # or
            # img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

            #overlay header , sidebar to webcam
            img[0:80, 0:1280] = overlay
            img[80:720, 1200:1280] = sidebar

            # Display current tool and color info
            cv2.putText(img, f"Tool: {selectedTool}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(img, f"Color: {selectedColor}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if tool == "Eraser":
                cv2.putText(img, f"Thickness: {eraserThickness}", (20,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            else:
                cv2.putText(img, f"Thickness: {brushThickness}", (20,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                
                
            # Display voice command feedback
            if time.time() - last_command_time < command_display_duration and command_display:
                cv2.putText(img, command_display, (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Display voice status
            if voice_listener and voice_listener.running:
                cv2.putText(img, "Voice: ON", (1150, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                cv2.putText(img, "Voice: OFF", (1150, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # cv2.putText(img, "Thickness: Thumb-Index pinch to adjust", (800, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # # showing frame
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        # Clean up resources
        if voice_listener:
            voice_listener.stop_listening()
        cap.release()
        cv2.destroyAllWindows()
   
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/board")
def board():
    return render_template("video.html")

@app.route("/video")
def video():
    # return Response(generate_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
    
if __name__== "__main__":
    print("Initializing voice command system...")
    app.config['mics'] = VoiceCommandListener.list_microphones()
    selected_mic = 2  # Brute use device 2

    # Validate if index 2 exists
    valid = any(mic[0] == 2 for mic in app.config['mics'])
    if not valid:
        print(f"Microphone index 2 not found. Changing to index 1")
        selected_mic = 1  # Fallback to default
    app.config['selected_mic'] = selected_mic
    app.run(host='0.0.0.0', port=5000, debug=False)
    