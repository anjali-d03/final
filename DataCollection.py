import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes =48# 0:'A' , 1:'B' , 2:'C' , 3:'C' ,4:'E' ,5:'R'
                         # , 6 :'S' , 7:'L' ,8:'D' , 9:'F' ,
#                            10 :'G' , 11:'K' , 12 :'M' , 13 :'N' ,14:'O'
#                            #15:'P' , 16:'Q' ,17:'T' ,18:'U' ,19:'V'
#
#                             20 :'W' ,21:'X' ,22:'Y',23:'J',24:'H',25:'Z'
#
#                             26 : '1' , 27 :'2' ,28:'3' ,29 :'4' ,30:'5'
#
#                             31:'6' ,32:'7' ,33:'8' ,34:'9' ,35:'10'
#
#                             36:'See you later' ,37:'thank you' ,38:'goodbye' ,39:'no'
#
#                             40 : 'Hello' , 41:'U' , 42:'Where', 43:'nice', 44:'need', 45:'yes'
#
#                             46:'Please' ,47:'like'}


dataset_size = 300

cap = cv2.VideoCapture(0)
for j in range(47,number_of_classes):

    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "s" to save ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
