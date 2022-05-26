from predictor import Predictor
import numpy as np
import cv2

predictor = Predictor("../models/other_models/model_epoch_48_98.6_final.h5")

cam = cv2.VideoCapture(0)
img_counter = 0
img_text = ['','']

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    imcrop = img[102:298, 427:623]

    output = np.ones((150, 150, 3)) * 255 #imagem 150x150, com fundo branco e 3 canais para as cores

    cv2.putText(output, str(img_text[1]), (15, 130), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0))
    
    cv2.imshow("ROI", imcrop)
    cv2.imshow("FRAME", frame)
    cv2.imshow("PREDICT", output)
   
    imggray = cv2.cvtColor(imcrop,cv2.COLOR_BGR2GRAY)
    
    img_name = "../temp/img.png"
    save_img = cv2.resize(imggray, (predictor.image_x, predictor.image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor.predict(img_name)
    print(str(img_text[0]))

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()