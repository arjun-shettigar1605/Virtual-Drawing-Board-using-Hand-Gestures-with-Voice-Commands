
#importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from voice import VoiceCommandListener


cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
cap.set(3,1280) #width=1280px
cap.set(4,720)  #height=720px
cap.set(10,150) #brightness=150%


#Canvas
canvas=np.zeros((720,1280,3),np.uint8)
canvas[:,:,:]=255
# Black Canvas
canvasBlack=np.zeros((720,1280,3),np.uint8)





#header bar image
overlay=cv2.imread("images/BarUp.png")[0:80,0:1280]

#SideBar image
sidebar=cv2.imread("images/BarSide.png")[80:720,1200:1280]




#Mediapipe hand object
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()  #hands=mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)

#Mediapipes Drawing tool for connecting hand landmarks
mp_draw=mp.solutions.drawing_utils


#tools
drawColor=(0,0,255)

# for displaying on screen
selectedColor='Blue'
selectedTool='Draw'

tool="Draw"  #important for selection

xp,yp=0,0           #previous position of index finger



# variable for drawing circle
countCircle=1

# variable for drawing rectangle
countRectangle=1

voice_listener = VoiceCommandListener()
voice_listener.start_listening()

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






while cap.isOpened():
    success,img=cap.read()
    if not success:
        print("Failed to capture frame from webcam")
        continue 
    #flipping to make correct aligned(1=horizontally)
    img=cv2.flip(img,1)


    # 2. Landmark and position finding
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # bcz mediapipe prefer RGB as it is trained in RGB
    results=hands.process(imgRGB)               # Hand Detected

    #IT IS IMPORTANT THAT IT MUST BE PLACED INSIDE LOOP
    landMark = []   #Landmark list for storing position of each finger                   ERROR 1 CAUSED

    try:
    # Check for voice commands (non-blocking)
        command = voice_listener.get_command()
        if command == "save":
            print("Saving the canvas...")
            # Save the canvas to a file with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phantom_canvas_{timestamp}.png"
            cv2.imwrite(filename, canvas)
            print(f"Canvas saved as {filename}")
        elif command == "clear":
            print("Clearing the canvas...")
            canvasBlack = np.zeros((720, 1280, 3), np.uint8)
            canvas[:, :, :] = 255
    
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
       # print(xm,ym)

        #3.opened fingers
        fingerList=fingerUp(landMark)

        # 4. Selection Mode=================================================================================
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
            if yi < 85:
                # check if fingers are in which x position

                #Red Color
                if 342 < xm < 480:
                    drawColor = (0, 0, 255)

                    selectedColor = "Red"

                #Blue Color
                elif 576 < xm < 720:
                    drawColor = (255, 100, 0)
                    selectedColor = "Blue"

                #Green Color
                elif 810 < xm < 960:
                    drawColor = (0, 255, 0)
                    selectedColor = "Green"


                #Eraser
                elif 1020 < xi < 1200:
                    drawColor = (0, 0, 0)
                    selectedColor = "none"
                    selectedTool='Eraser'
                    tool="Eraser"
            
            #side tool selection
            if xm>1220:
                if 81<ym<167:
                    #print("Clear all")
                    canvasBlack = np.zeros((720, 1280, 3), np.uint8)
                    canvas[:, :, :] = 255
                elif 192<ym<294:
                    #print("Draw tool")
                    tool="Draw"
                elif 320<ym<408:
                    #print("Circle tool")
                    tool="Circle"
                elif 440<ym<550:
                    #print("Rectangle")
                    tool="Rectangle"






        #5. Drawing Mode==================================================================================
        if fingerList[1] and fingerList[2]==0:

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
            if tool=="Draw":
                #when frame start dont make a line from 0,0 so draw a line from xi,yi to xi,yi ie a point
                if xp==0 and yp==0:
                    xp,yp=xi,yi

                # it is to automatically make eraser back to normal size
                if drawColor!=(0,0,0):
                    cv2.line(img,(xp,yp),(xi,yi),drawColor,10)
                    cv2.line(canvas,(xp,yp),(xi,yi),drawColor,10)
                    cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, 10)
                else:
                    cv2.line(img, (xp, yp), (xi, yi), drawColor, 70)
                    cv2.line(canvas, (xp, yp), (xi, yi), (255,255,255), 70)
                    cv2.line(canvasBlack, (xp, yp), (xi, yi), drawColor, 70)
                #update xp and yp
                xp,yp=xi,yi

            elif tool=="Circle":
                if countCircle==1:
                    xstart_circle,ystart_circle=xi,yi
                    countCircle=0
                cv2.circle(img,(xstart_circle,ystart_circle), int(((xstart_circle - xi) ** 2 + (ystart_circle - yi) ** 2) ** 0.5),drawColor,10)
                xlast_circle,ylast_circle=xi,yi
            elif tool == "Rectangle":
                if countRectangle == 1:
                    xstart_rect, ystart_rect = xi, yi
                    countRectangle = 0
                cv2.rectangle(img, (xstart_rect, ystart_rect),(xi,yi), drawColor, 10)
                xlast_rect, ylast_rect = xi, yi




    #6 . Adding canvas and real fram


    imgGray=cv2.cvtColor(canvasBlack,cv2.COLOR_BGR2GRAY)
    _,imgBin=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgBin=cv2.cvtColor(imgBin,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgBin)
    img=cv2.bitwise_or(img,canvasBlack)


    #or
    #img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)


    #trying to overlay header , sidebar to webcam
    img[0:80,0:1280]=overlay
    img[80:720,1200:1280]=sidebar




    #showing frame
    cv2.imshow("Phantom- White Board",canvas)
    cv2.imshow("Phantom - A Virtual Board",img)

    #exit condition by using esc
    if cv2.waitKey(1)==27:
        break
