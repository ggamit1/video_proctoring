from imutils.video import FPS
import numpy as np
import imutils
import cv2
import dlib 
from imutils import face_utils
from datetime import datetime


detector = dlib.get_frontal_face_detector() 
# need to dlib face_landmarks.dat model for detect face
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
stream = cv2.VideoCapture(0)
fps = FPS().start()
# Loop over frames from the video file stream
i=0
last_save_time = datetime.now()
while True:
    
    # Grab the frame from the threaded video file stream
    (grabbed, frame) = stream.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
   
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    faces = faceCascade.detectMultiScale(
            frame,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
    
    current_time = datetime.now()
    diffrene = current_time - last_save_time
    if len(faces)>1  and diffrene.seconds > 10:
        rects = detector(frame, 1) 
        if len(rects)>1:
            i+=1
            cv2.imwrite(f"proctoring{i}.png",frame)
            last_save_time = current_time
        
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # roi_gray = frame[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w] 
        # roi_image = frame[y:y+h,x:x+w]
        # blur = cv2.blur(roi_image, (21 ,21))
        # frame[y:y+h, x:x+w] = blur
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    fps.update()
 # stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
stream.release()
cv2.destroyAllWindows()

