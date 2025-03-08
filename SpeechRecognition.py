import speech_recognition as sr

def test_microphone():
    r = sr.Recognizer()
    
    # List all available microphones
    mics = sr.Microphone.list_microphone_names()
    print("Available microphones:")
    for i, mic_name in enumerate(mics):
        print(f"{i}: {mic_name}")
    
    # Allow user to select a microphone
    selected_mic = None
    while selected_mic is None:
        try:
            mic_index = int(input("Enter the microphone index to use: ").strip())
            if 0 <= mic_index < len(mics):
                selected_mic = mic_index
            else:
                print("Invalid index. Please enter a number from the list.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Use the selected microphone
    with sr.Microphone(device_index=selected_mic) as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=2)
        print("Say something!")
        
        try:
            audio = r.listen(source, timeout=10)
            print("Recognizing...")
            result = r.recognize_google(audio)
            print(f"You said: {result}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_microphone()
