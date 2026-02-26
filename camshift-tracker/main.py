import cv2
import numpy as np

# Open camera (macOS backend)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Kalman Filter (x, y, dx, dy)
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03


drawing = False
ix, iy = -1, -1
rectangle = None
roi_hist = None
track_window = None

# Mouse selection
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rectangle = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangle = (ix, iy, x, y)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", draw_rectangle)

# MeanShift stopping criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    temp = frame.copy()

    # Draw selection box before learning
    if rectangle is not None and roi_hist is None:
        x1, y1, x2, y2 = rectangle
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        cv2.rectangle(temp,(x1,y1),(x2,y2),(0,255,0),2)

    key = cv2.waitKey(1) & 0xFF

    # Learn object when ENTER pressed
    if key == 13 and rectangle is not None:
        x1, y1, x2, y2 = rectangle
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)

        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # remove low color / dark pixels
        mask = cv2.inRange(hsv_roi, (0,60,32), (180,255,255))

        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        track_window = (x1, y1, x2-x1, y2-y1)
        print("Tracking started!")

    # -------- MeanShift Tracking --------
    if roi_hist is not None:

        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        pts = cv2.boxPoints(ret)
        pts = pts.astype(int)
        cv2.polylines(temp,[pts],True,(255,0,0),2)



    cv2.imshow("Camera", temp)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
