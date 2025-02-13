import cv2
from text_emotion import detect_text_emotion
from facial_emotion import detect_facial_emotion

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        facial_emotion = detect_facial_emotion(frame)
        print(f"Facial Emotion: {facial_emotion}")

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
