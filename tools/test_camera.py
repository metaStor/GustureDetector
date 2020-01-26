import cv2
import time


cap = cv2.VideoCapture(1)
# 准备时间
time.sleep(1.88)
while True:
    # get a image
    _, image = cap.read()
    # show a frame
    cv2.imshow("Camera", image)
    # quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
