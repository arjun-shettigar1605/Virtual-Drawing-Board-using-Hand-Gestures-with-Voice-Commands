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
cap.set(3, 1280)  # (3)width=1280px
cap.set(4, 720)   # (4)height=720px
cap.set(10, 150)  # brightness=150%

# No use of show_window() function as the video is being displayed in the browser, and this function works only in the local system.

# def show_window():
#     global last_command_time
#     global command_display
#     while True:
#         cv2.imshow("Phantom- White Board", canvas)
#         cv2.imshow("Phantom - A Virtual Board", img)

#         # exit condition by using esc
#         key = cv2.waitKey(1)
#         if key == 27:  # ESC key
#             break
#         elif key == ord('v'):  # Toggle voice recognition with 'v' key
#             if voice_listener:
#                 if voice_listener.running:
#                     voice_listener.stop_listening()
#                     command_display = "Voice commands disabled"
#                 else:
#                     voice_listener.start_listening()
#                     command_display = "Voice commands enabled"
#                 last_command_time = time.time()                
#         elif key == ord('+'):  # Manually increase thickness with '+' key
#             if tool == "Eraser":
#                 eraserThickness = min(eraserThickness + thicknessStep * 2, maxEraserThickness)
#                 command_display = f"Eraser thickness: {eraserThickness}"
#             else:
#                 brushThickness = min(brushThickness + thicknessStep, maxThickness)
#                 command_display = f"Brush thickness: {brushThickness}"
#             last_command_time = time.time()
#         elif key == ord('-'):  # Manually decrease thickness with '-' key
#             if tool == "Eraser":
#                 eraserThickness = max(eraserThickness - thicknessStep * 2, minThickness * 2)
#                 command_display = f"Eraser thickness: {eraserThickness}"
#             else:
#                 brushThickness = max(brushThickness - thicknessStep, minThickness)
#                 command_display = f"Brush thickness: {brushThickness}"
#             last_command_time = time.time()
#     cv2.destroyAllWindows()
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
    overlay_path = os.path.join(static_folder, 'images', 'BarSide2.png')
    # # SideBar image
    overplay_img = cv2.imread(overlay_path)
    # sidebar = cv2.imread("images/sam3.png")[80:720, 1200:1280]

    if overplay_img is None:
        print(f"Error: Could not load image from {overlay_path}")
        overlay = np.zeros((80, 1280, 3), np.uint8)
    else:
        overlay = overplay_img[0:80, 0:1280]    
        
    sidebar_path = os.path.join(static_folder, 'images', 'BarSide2.png')
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

    # variable for circle
    countCircle = 1
    # variable for rect
    countRectangle = 1

    # Pointer thickness control variavles

    brushThickness = 10
    eraserThickness = 70 
    minThickness = 2
    maxThickness = 40
    maxEraserThickness = 100
    thicknessStep = 3 #step increase during voice command increase method
    minDistance = 20
    maxDistance = 150



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

                # 4. Selection Mode=================================================================================
                if fingerList[1] and fingerList[2]:

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

                    # to make discontinuity after selection
                    xp, yp = 0, 0

                    cv2.rectangle(img, (xi - 10, yi - 15), (xm + 10, ym + 20), drawColor, -1)

                    # check if finger on header portion   check later y 125 not 80
                    if yi < 85:
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
                        elif 300 < xm < 420:
                            drawColor = (0, 255, 0)
                            selectedColor = "Green"

                        # Yellow Color
                        elif 435 < xm < 570:
                            drawColor = (0, 255, 255)
                            selectedColor = "Yellow"
                            
                        # Violet Color
                        elif 580 < xm < 715:
                            drawColor = (211, 0, 148)
                            selectedColor = "Violet"
                        
                        # Pink Color:
                        elif 780 < xm < 890:
                            drawColor = (203, 192, 255)
                            selectedColor = "Pink"
                            
                        # Black Color:
                        elif 925 < xm < 1055:
                            drawColor = (0,0,0)
                            selectedColor = "Black"
                        
                        # Eraser
                        elif 1080 < xm < 1120:
                            drawColor = (0,0,0)
                            selectedColor = "none"
                            selectedTool = 'Eraser'
                            tool = "Eraser"
                        

                    # side tool selection
                    if xm > 1220:
                        if 81 < ym < 167:
                            # print("Clear all")
                            canvasBlack = np.zeros((720, 1280, 3), np.uint8)
                            canvas[:, :, :] = 255
                            command_display = "Canvas cleared"
                            last_command_time = time.time()
                        elif 192 < ym < 294:
                            # print("Draw tool")
                            tool = "Draw"
                            selectedTool = 'Draw'
                            command_display = "Draw tool selected"
                            last_command_time = time.time()
                        elif 320 < ym < 408:
                            # print("Circle tool")
                            tool = "Circle"
                            selectedTool = 'Circle'
                            command_display = "Circle tool selected"
                            last_command_time = time.time()
                        elif 440 < ym < 550:
                            # print("Rectangle")
                            tool = "Rectangle"
                            selectedTool = 'Rectangle'
                            command_display = "Rectangle tool selected"
                            last_command_time = time.time()

                # 5. Drawing Mode==================================================================================
                if fingerList[1] and fingerList[2] == 0:

                    
                    current_thickness = brushThickness
                    if fingerList[0] == 1 and len(landMark) > 8:
                        thumb_index_distance = calculate_distance((xthumb, ythumb), (xi, yi))
                        if tool == "Eraser":
                            mapped_thickness = max(minThickness * 2, min(maxEraserThickness, minThickness * 2 + (maxEraserThickness - minThickness * 2) * ((thumb_index_distance - minDistance) / (maxDistance - minDistance))))
                            eraserThickness = int(mapped_thickness)
                        else:
                            mapped_thickness = max(minThickness, min(maxThickness, minThickness + (maxThickness - minThickness) * ((thumb_index_distance - minDistance) / (maxDistance - minDistance))))
                            brushThickness = int(mapped_thickness)
                    
                        cv2.putText(img, f"Thickness: {int(mapped_thickness)}", (xi + 20, yi), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                        # Draw a line between thumb and index to visualize the measurement
                        cv2.line(img, (xthumb, ythumb), (xi, yi), (0, 255, 0), 2)
                        
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

                    elif tool == "Circle":
                        if countCircle == 1:
                            xstart_circle, ystart_circle = xi, yi
                            countCircle = 0
                        cv2.circle(img, (xstart_circle, ystart_circle), int(((xstart_circle - xi) ** 2 + (ystart_circle - yi) ** 2) ** 0.5), drawColor, brushThickness)
                        xlast_circle, ylast_circle = xi, yi
                    elif tool == "Rectangle":
                        if countRectangle == 1:
                            xstart_rect, ystart_rect = xi, yi
                            countRectangle = 0
                        cv2.rectangle(img, (xstart_rect, ystart_rect), (xi, yi), drawColor, brushThickness)
                        xlast_rect, ylast_rect = xi, yi

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

            cv2.putText(img, "Thickness: Thumb-Index pinch to adjust", (800, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # showing frame
            
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
    