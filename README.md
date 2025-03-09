# **Virtual Drawing Board Using Hand Gesture Recognition and Voice Command Integration**  
This project is a **Virtual Drawing Board** that enables users to interact with a digital canvas using **hand gestures and voice commands** without requiring physical contact. It combines **Computer Vision** and **Speech Recognition** technologies to create an intuitive, contactless drawing experience.

---

## 🔑 **Project Overview**  
The Virtual Drawing Board leverages **MediaPipe** for real-time hand gesture recognition and **OpenCV** to process live video feeds from the webcam. The application can track hand landmarks to interpret various drawing actions like sketching, erasing, and selecting tools. Additionally, voice commands using the **SpeechRecognition** module allow users to clear the canvas, save their artwork, and adjust pointer thickness through simple spoken instructions.

---

## **Features**  
- Contactless drawing with hand gestures  
- Voice commands to perform actions like:
    - **"Clear"** – Erase the canvas  
    - **"Screenshot"** – Save the artwork as an image 
    - **"Increase"** – Thicker brush  
    - **"Decrease"** – Thinner brush  
- Real-time hand tracking using **MediaPipe**  
- Adjustable brush thickness with gesture controls  
- Circle and rectangle shape drawing  
- Color and tool selection  

---

## 🎯 **Tech Stack Used**  
- Python  
- Flask (Backend Framework)  
- MediaPipe (Hand Gesture Recognition)  
- OpenCV (Computer Vision)  
- SpeechRecognition (Voice Command Processing)  
- HTML/CSS  
- JavaScript  

---

## 📌 **Folder Structure**  
```
Virtual Drawing Board/
│
├─ static/           # CSS, JS, and images
├─ templates/        # HTML files
├─ App2.py           # Main Flask backend file
└─ requirements.txt  # Dependencies
```

---

## **How to Run the Project**  
1. Clone the repository:  
```bash
git clone https://github.com/username/Virtual-Drawing-Board-using-Hand-Gestures-with-Voice-Commands.git
cd Virtual-Drawing-Board-using-Hand-Gestures-with-Voice-Commands
```
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
3. Run the application:  
```bash
python App2.py
```
4. Visit:  
```
http://127.0.0.1:5000/
```

---

## **Screenshots**  
Below are the screenshots of the user interface, the Canvas Template, and the Tools and Hand Gestures in use.

![sam1](https://github.com/user-attachments/assets/522120b5-e3d3-48af-8bc4-bf689cd36383)
![sam2](https://github.com/user-attachments/assets/c35c2c32-a303-418e-926f-1b0b73082d18)
![sam3](https://github.com/user-attachments/assets/369ea965-9747-44a4-acfb-bb40a430ddc5)
![Screenshot 2025-03-08 133705](https://github.com/user-attachments/assets/7bca4c65-962f-4fe3-ab35-4c52d7391c54)
![Screenshot 2025-03-08 133741](https://github.com/user-attachments/assets/cf5873da-fd5b-4788-a645-25404b9dd80f)
![Screenshot 2025-03-08 202508](https://github.com/user-attachments/assets/b3c3263c-7165-40a7-a038-5470cd6fb791)


---

## 🎯 **Future Scope**  
- Multi-hand gesture support  
- Advanced shape recognition  
- Multi-user collaboration  
- Cloud storage integration  
- Background noise filtering in voice commands  


---

## ⭐ **Feel free to contribute and make this project better!**
