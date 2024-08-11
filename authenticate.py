import cv2
from model import getFaceCascade, getRecognitionModel
import time

def main():
    face_cascade = getFaceCascade()
    recognizer = getRecognitionModel()
    recognizer.read('/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Weights/face_recognizer_model.yml')

    video_capture = cv2.VideoCapture(0)
    print("Opening CV")
    
    while True:
        ret, frame = video_capture.read()
        time.sleep(0.1)  # Add a delay between frame captures

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
            if label == 1 and confidence <70:
                print("Accepted")
                video_capture.release()
                cv2.destroyAllWindows()
                return  # Exit the program
            else:
                print("Retrying...")
        
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
