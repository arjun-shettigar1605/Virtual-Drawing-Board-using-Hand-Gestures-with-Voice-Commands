#importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import speech_recognition as sr
import time
import pyaudio
from datetime import datetime

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
        print("Available microphones:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            # Only include input devices (microphones)
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
            print(f"Voice command listener started using device index {self.device_index}. Say 'save' or 'clear'.")
    
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
            
            while self.running:
                try:
                    with self.microphone as source:
                        print("Listening for commands...")
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                    
                    try:
                        # Recognize speech using Google Web Speech API
                        command = self.recognizer.recognize_google(audio).strip().lower()
                        print(f"Detected: {command}")
                        
                        # Check for valid commands
                        if "save" in command:
                            self.command_queue.put("save")
                        elif "clear" in command:
                            self.command_queue.put("clear")
                        # Add more commands as needed
                    
                    except sr.UnknownValueError:
                        # Speech wasn't understood
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

# Print available microphones and ask for selection
print("Initializing voice command system...")
mics = VoiceCommandListener.list_microphones()

selected_mic = None
if len(mics) > 0:
    print("\nSelect a microphone by entering its device number:")
    try:
        selected_mic = int(input("Enter device number (or press Enter for default): ").strip())
        # Validate the selection
        valid_selection = False
        for mic_index, _ in mics:
            if selected_mic == mic_index:
                valid_selection = True
                break
        
        if not valid_selection:
            print(f"Invalid selection. Using default microphone.")
            selected_mic = None
    except ValueError:
        print("Using default microphone.")
        selected_mic = None
else:
    print("No microphones detected. Voice commands will be disabled.")

# webcam initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
cap.set(3, 1280)  # (3)width=1280px
cap.set(4, 720)   # (4)height=720px
cap.set(10, 150)  # brightness=150%

# Canvas
canvas = np.zeros((720, 1280, 3), np.uint8)
canvas[:, :, :] = 255
# Black Canvas
canvasBlack = np.zeros((720, 1280, 3), np.uint8)

# header bar image
try:
    overlay = cv2.imread("images/BarSide2.png")[0:80, 0:1280]
    # SideBar image
    sidebar = cv2.imread("images/BarSide2.png")[80:720, 1200:1280]
except Exception as e:
    print(f"Error loading UI images: {e}")
    # Create default UI elements
    overlay = np.zeros((80, 1280, 3), np.uint8)
    overlay[:, :] = (200, 200, 200)  # Gray color
    
    sidebar = np.zeros((640, 80, 3), np.uint8)
    sidebar[:, :] = (200, 200, 200)  # Gray color

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
if mics:
    voice_listener = VoiceCommandListener(device_index=selected_mic)
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
                    if command == "save":
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
                    elif command == "thicker":
                        if tool == "Eraser":
                            eraserThickness = min(eraserThickness + thicknessStep*2, maxEraserThickness)
                            command_display = f"Eraser thickness: {eraserThickness}"
                        else:
                            brushThickness = min(brushThickness + thicknessStep, maxThickness)
                            command_display = f"Brush thickness: {brushThickness}"
                        last_command_time=time.time()
                    elif command == "thinner":
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
        cv2.putText(img, f"Tool: {selectedTool}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"Color: {selectedColor}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if tool == "Eraser":
            cv2.putText(img, f"Thickness: {eraserThickness}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        else:
            cv2.putText(img, f"Thickness: {brushThickness}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            
        # Display voice command feedback
        if time.time() - last_command_time < command_display_duration and command_display:
            cv2.putText(img, command_display, (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Display voice status
        if voice_listener and voice_listener.running:
            cv2.putText(img, "Voice: ON", (1150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.putText(img, "Voice: OFF", (1150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(img, "Thickness: Thumb-Index pinch to adjust", (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # showing frame
        cv2.imshow("Phantom- White Board", canvas)
        cv2.imshow("Phantom - A Virtual Board", img)

        # exit condition by using esc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == ord('v'):  # Toggle voice recognition with 'v' key
            if voice_listener:
                if voice_listener.running:
                    voice_listener.stop_listening()
                    command_display = "Voice commands disabled"
                else:
                    voice_listener.start_listening()
                    command_display = "Voice commands enabled"
                last_command_time = time.time()
        elif key == ord('m'):  # Change microphone with 'm' key
            if voice_listener:
                # Stop current listener
                voice_listener.stop_listening()
                
                # List microphones and get selection
                mics = VoiceCommandListener.list_microphones()
                if mics:
                    try:
                        selected_mic = int(input("Enter device number (or press Enter for default): ").strip())
                        voice_listener.set_microphone(selected_mic)
                        voice_listener.start_listening()
                        command_display = f"Microphone changed to device {selected_mic}"
                    except ValueError:
                        voice_listener.set_microphone(None)  # Default microphone
                        voice_listener.start_listening()
                        command_display = "Using default microphone"
                    last_command_time = time.time()
                    
        elif key == ord('+'):  # Manually increase thickness with '+' key
            if tool == "Eraser":
                eraserThickness = min(eraserThickness + thicknessStep * 2, maxEraserThickness)
                command_display = f"Eraser thickness: {eraserThickness}"
            else:
                brushThickness = min(brushThickness + thicknessStep, maxThickness)
                command_display = f"Brush thickness: {brushThickness}"
            last_command_time = time.time()
        elif key == ord('-'):  # Manually decrease thickness with '-' key
            if tool == "Eraser":
                eraserThickness = max(eraserThickness - thicknessStep * 2, minThickness * 2)
                command_display = f"Eraser thickness: {eraserThickness}"
            else:
                brushThickness = max(brushThickness - thicknessStep, minThickness)
                command_display = f"Brush thickness: {brushThickness}"
            last_command_time = time.time()

finally:
    # Clean up resources
    if voice_listener:
        voice_listener.stop_listening()
    cap.release()
    cv2.destroyAllWindows()