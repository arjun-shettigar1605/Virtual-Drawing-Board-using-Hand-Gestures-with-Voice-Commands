# **Virtual Drawing Board Using Hand Gesture Recognition and Voice Command Integration (VisionBoard)**  
This project is a **Virtual Drawing Board** that enables users to interact with a digital canvas using **hand gestures and voice commands** without requiring physical contact. It combines **Computer Vision** and **Speech Recognition** technologies to create an intuitive, contactless drawing experience.

---

## 🔑 **Project Overview**  
The Virtual Drawing Board leverages **MediaPipe** for real-time hand gesture recognition and **OpenCV** to process live video feeds from the webcam. The application can track hand landmarks to interpret various drawing actions like sketching, erasing, and selecting tools. Additionally, voice commands using the **SpeechRecognition** module allow users to clear the canvas, save their artwork, and adjust pointer thickness through simple spoken instructions.

---

## **Features**  
- Contactless drawing with hand gestures  
- Voice commands to perform actions like:
    - **"Clear"** – Erase the canvas  
    - **"Save"** – Save the artwork as an image  
    - **"Increase Thickness"** – Thicker brush  
    - **"Decrease Thickness"** – Thinner brush  
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
├─ static/        # CSS, JS, and images
├─ templates/     # HTML files
├─ Phantom3.py    # Main Flask backend file
└─ requirements.txt  # Dependencies
```

---

## **How to Run the Project**  
1. Clone the repository:  
```bash
git clone https://github.com/your_username/Virtual-Drawing-Board.git
cd Virtual-Drawing-Board
```
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
3. Run the application:  
```bash
python Phantom3.py
```
4. Visit:  
```
http://127.0.0.1:5000/
```

---

## **Screenshots**  
![image](https://github.com/user-attachments/assets/206ff101-1544-4be9-97a6-bf662949e43f)
![image](https://github.com/user-attachments/assets/5c11ea82-9495-416d-9d0b-123cb307a79c)
![sam3](https://github.com/user-attachments/assets/61d6a0db-cd26-44d4-a347-c09b7b84befb)

 

---

## 🎯 **Future Scope**  
- Multi-hand gesture support  
- Advanced shape recognition  
- Multi-user collaboration  
- Cloud storage integration  
- Background noise filtering in voice commands  

---


## ⭐ **Feel free to contribute and make this project better!**
