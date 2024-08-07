import cv2

from deepface import DeepFace

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:

    a,b=cap.read()
   
    gray=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

    faces=face_cascade.detectMultiScale(gray,1.3,7)

    for(x,y,w,h) in faces:

        cv2.rectangle(b,(x,y),(x+w,y+h),(0,255,0),5)

        face_recog=rgb[y:y+h,x:x+w]

        result=DeepFace.analyze(face_recog,actions=['emotion'],enforce_detection=False)

        emotion=result[0]['dominant_emotion']

        cv2.putText(b,emotion,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

    cv2.imshow('Face',b)

    if cv2.waitKey(1)==ord('q'):

        break

cap.release()

cv2.destroyAllWindows()    
