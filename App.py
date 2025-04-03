#importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np


# video capture object created with webcam no 0
cap=cv2.VideoCapture(0)
cap.set(3,1280) #width=1280px
cap.set(4,720)  #height=720px
cap.set(10,150) #brightness=150%


#Canvas
canvas=np.zeros((720,1280,3),np.uint8)
canvas[:,:,:]=255
# Black Canvas
canvasBlack=np.zeros((720,1280,3),np.uint8)


# Load header bar image
overlay_path = "images/sam3.png"
overlay = cv2.imread(overlay_path)
if overlay is None:
    print(f"Error: Could not load image from {overlay_path}")
    overlay = np.zeros((80, 1280, 3), np.uint8)  # Fallback to a black bar
else:
    overlay = overlay[0:80, 0:1280]

# Load sidebar image
sidebar_path = "images/sam3.png"
sidebar = cv2.imread(sidebar_path)
if sidebar is None:
    print(f"Error: Could not load image from {sidebar_path}")
    sidebar = np.zeros((640, 80, 3), np.uint8)  # Fallback to a black sidebar
else:
    sidebar = sidebar[80:720, 1200:1280]
#Mediapipe hand object
mp_hands=mp.solutions.hands
hands=mp_hands.Hands() 

#Mediapipes Drawing tool for connecting h`and landmarks
mp_draw=mp.solutions.drawing_utils


#tools
drawColor=(0,0,255)

# for displaying on screen
selectedColor='Blue'
selectedTool='Draw'

tool="Draw"  #important for selection

xp,yp=0,0    #previous position of index finger


# variable for drawing circle
countCircle=1

# variable for drawing rectangle
countRectangle=1
# Pointer thickness control variables
brushThickness = 10
eraserThickness = 70
minThickness = 2
maxThickness = 40
maxEraserThickness = 100
minDistance = 20
maxDistance = 150

#Function for finding how much fingers are up
tipIds=[8,12,16,20]  # finger tip ids except for thump tip (4)
def fingerUp(landmark):
    fingerList=[]
    #thump up/down finding is different if thumptip(4) is left to id 3 then up else down ie, x(id4)<x(id3)
    if landmark[4][1]<landmark[3][1]:
        fingerList.append(1)                                    # 0-id 1-x 2-y in landmark
    else:
        fingerList.append(0)

    #For the rest of fingers if y(id-tip)<y(id-middlepart) then up else down (id-2 bcz middle part of finger)
    for id in tipIds:
        if landmark[id][2]<landmark[id-2][2]:
            fingerList.append(1)
        else:
            fingerList.append(0)

    return fingerList

def calculate_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

try:
    while cap.isOpened():
        success,img=cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            continue
        # Flip the image horizontally for a mirror effect
        img=cv2.flip(img,1)


        # 2. Landmark and position finding
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # bcz mediapipe prefer RGB as it is trained in RGB
        results=hands.process(imgRGB)               # Hand Detected
        
        landMark = []   #Landmark list for storing position of each finger


        #if hand is detected
        if results.multi_hand_landmarks:
            lndmrk = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, lndmrk, mp_hands.HAND_CONNECTIONS)  # drawing connection not necessary

            for id, lm in enumerate(lndmrk.landmark):
                height, width, _ = img.shape
                # this is done because lm.x gives ratio of x position but we need pixel value so multiply by width , same for height
                x, y = int(lm.x * width), int(lm.y * height)
                # appending each landmarks id and position as a list to landMark
                landMark.append([id, x, y])

            xi,yi=landMark[8][1:]       #index fingers position
            xm,ym=landMark[12][1:]      #middle fingers position
            
            if len(landMark) > 4:
                xthumb, ythumb = landMark[4][1:]  # Thumb position for thickness control
        

            # opened fingers
            fingerList=fingerUp(landMark)
            
            #4. Selection Mode
            if fingerList[1] and fingerList[2]:

                #circle finishing
                if countCircle==0:
                    cv2.circle(img, (xstart_circle, ystart_circle),int(((xstart_circle - xlast_circle) ** 2 + (ystart_circle - ylast_circle) ** 2) ** 0.5), drawColor, 10)
                    cv2.circle(canvas, (xstart_circle, ystart_circle),int(((xstart_circle - xlast_circle) ** 2 + (ystart_circle - ylast_circle) ** 2) ** 0.5),drawColor, 10)
                    cv2.circle(canvasBlack, (xstart_circle, ystart_circle),int(((xstart_circle - xlast_circle) ** 2 + (ystart_circle - ylast_circle) ** 2) ** 0.5),drawColor, 10)
                    countCircle=1

                #rectangle finishing
                if countRectangle==0:
                    cv2.rectangle(img, (xstart_rect, ystart_rect), (xlast_rect, ylast_rect), drawColor, 10)
                    cv2.rectangle(canvas, (xstart_rect, ystart_rect), (xlast_rect, ylast_rect), drawColor, 10)
                    cv2.rectangle(canvasBlack, (xstart_rect, ystart_rect), (xlast_rect, ylast_rect), drawColor, 10)
                    countRectangle=1


                #to make discontinuity after selection
                xp,yp=0,0

                cv2.rectangle(img,(xi-10,yi-15),(xm+10,ym+20),drawColor,-1)

                #check if finger on header portion   check later y 125 not 80
                if yi < 85:  # Header area
                    if 20 < xm < 142:  # Red
                        drawColor = (0, 0, 255)
                        selectedColor = "Red"
                    elif 160 < xm < 280:  # Blue
                        drawColor = (255, 100, 0)
                        selectedColor = "Blue"
                    elif 300 < xm < 420:  # Green
                        drawColor = (0, 255, 0)
                        selectedColor = "Green"
                    elif 435 < xm < 570:  # Yellow
                        drawColor = (0, 255, 255)
                        selectedColor = "Yellow"
                    elif 580 < xm < 715:  # Violet
                        drawColor = (211, 0, 148)
                        selectedColor = "Violet"
                    elif 780 < xm < 890:  # Pink
                        drawColor = (203, 192, 255)
                        selectedColor = "Pink"
                    elif 925 < xm < 1055:  # Black
                        drawColor = (0, 0, 0)
                        selectedColor = "Black"
                    elif 1080 < xm < 1120:  # Eraser
                        drawColor = (0, 0, 0)
                        selectedColor = "none"
                        selectedTool = 'Eraser'
                        tool = "Eraser"
                
                #side tool selection
                if xm > 1220:
                    if 81 < ym < 167:  # Clear canvas
                        canvasBlack = np.zeros((720, 1280, 3), np.uint8)
                        canvas[:, :, :] = 255
                    elif 192 < ym < 294:  # Draw tool
                        tool = "Draw"
                        selectedTool = 'Draw'
                    elif 320 < ym < 408:  # Circle tool
                        tool = "Circle"
                        selectedTool = 'Circle'
                    elif 440 < ym < 550:  # Rectangle tool
                        tool = "Rectangle"
                        selectedTool = 'Rectangle'
                        

            #5. Drawing Mode
            if fingerList[1] and fingerList[2]==0:
                if fingerList[0] == 1 and len(landMark) > 8:
                    thumb_index_distance = calculate_distance((xthumb, ythumb), (xi, yi))
                    if tool == "Eraser":
                        mapped_thickness = max(minThickness * 2, min(maxEraserThickness, minThickness * 2 + (maxEraserThickness - minThickness * 2) * ((thumb_index_distance - minDistance) / (maxDistance - minDistance))))
                        eraserThickness = int(mapped_thickness)
                    else:
                        mapped_thickness = max(minThickness, min(maxThickness, minThickness + (maxThickness - minThickness) * ((thumb_index_distance - minDistance) / (maxDistance - minDistance))))
                        brushThickness = int(mapped_thickness)
                    cv2.putText(img, f"Thickness: {int(mapped_thickness)}", (xi + 20, yi), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.line(img, (xthumb, ythumb), (xi, yi), (0, 255, 0), 2)  # Visualize thumb-index distance
                #print("Drawing Mode")

                cv2.circle(img,(xi,yi),15,drawColor,-1)

                if tool=="Eraser":
                    #when frame start dont make a line from 0,0 so draw a line from xi,yi to xi,yi ie a point
                    if xp==0 and yp==0:
                        xp,yp=xi,yi

                    #it is to automatically make drawing back to normal size
                    if drawColor==(0,0,0):
                        cv2.line(img, (xp, yp), (xi, yi), drawColor, 70)
                        cv2.line(canvas, (xp, yp), (xi, yi), (255,255,255), 70)
                        cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, 70)
                    else:
                        cv2.line(img, (xp, yp), (xi, yi), drawColor, 10)
                        cv2.line(canvas, (xp, yp), (xi, yi), drawColor, 10)
                        cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, 10)
                    #update xp and yp
                    xp,yp=xi,yi


                #Drawing
                if tool == "Eraser":
                    if xp == 0 and yp == 0:
                        xp, yp = xi, yi
                    cv2.line(img, (xp, yp), (xi, yi), drawColor, eraserThickness)
                    cv2.line(canvas, (xp, yp), (xi, yi), (255, 255, 255), eraserThickness)
                    xp, yp = xi, yi

                elif tool == "Draw":
                    if xp == 0 and yp == 0:
                        xp, yp = xi, yi
                    cv2.line(img, (xp, yp), (xi, yi), drawColor, brushThickness)
                    cv2.line(canvas, (xp, yp), (xi, yi), drawColor, brushThickness)
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

        #6 . Adding canvas and real fram

        # Display current tool and color info
        cv2.putText(img, f"Tool: {selectedTool}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"Color: {selectedColor}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if tool == "Eraser":
            cv2.putText(img, f"Thickness: {eraserThickness}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.putText(img, f"Thickness: {brushThickness}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "Thickness: Thumb-Index pinch to adjust", (800, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Show the image in a window
        cv2.imshow("Virtual Board", img)

        # Exit on pressing 'ESC'
        if cv2.waitKey(1) == 27:
            break

finally:
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
