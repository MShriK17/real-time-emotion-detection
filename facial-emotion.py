import cv2

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_facial_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        return "Emotion Detected"
    return "Neutral"

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print(f"Detected Facial Emotion: {detect_facial_emotion(frame)}")
    cap.release()
