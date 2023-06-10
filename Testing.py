import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

engine = pyttsx3.init()

f = open(r"C:\Users\user\PycharmProjects\pythonProject\model.p", "rb")
model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'I', 4: 'E', 5: 'R', 6: 'S', 7: 'L', 8: 'D', 9: 'F', 10: 'G', 11: 'K', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'T', 18: 'U', 19: 'V', 20: 'W', 21: 'X', 22: 'Y', 23: 'J', 24: 'H',
               25: 'Z', 26: '1', 27: '2', 28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '10',
36:'See you later' ,37:'thank you' ,38:'goodbye' ,39:'no',
40 : 'Hello' ,  41:'You' , 42:'Where',43:'nice', 44:'need', 45:'yes',46:'Please',47:'like'}

word = ""  # Initialize an empty word
sentence = ""

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux[:42])])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Add predicted character to the word when 'y' key is pressed
        key = cv2.waitKey(1)
        if key == ord('y'):
            word += predicted_character

        if key == ord("s"):
            sentence += word + ' '
            word = ''
        elif key == ord("d"):
            word = word[:-1]

        if key == ord("r"):
            print(sentence)
            engine.say(sentence)
            engine.runAndWait()

        if key == ord("x"):
            sentence = ""

        # Display the predicted word and complete sentence on the video screen
        cv2.putText(frame, "Predicted Word: " + word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (21, 44, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Complete Sentence: " + sentence, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (45, 121, 255), 2, cv2.LINE_AA)

    #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('frame', 700, 750)  # Adjust the size as per your preference
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  # Press Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
