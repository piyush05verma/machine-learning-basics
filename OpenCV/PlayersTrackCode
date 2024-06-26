import cv2
import numpy as np

cap = cv2.VideoCapture('OpenCV/volleyball_match.mp4')

detector = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blur = cv2.GaussianBlur(frame, (5, 5), sigmaX = 35)
    yuv = cv2.cvtColor(blur, cv2.COLOR_RGB2YUV)
    mask_motion = detector.apply(blur)
    
    a_low = np.array([25, 180, 110])
    a_high = np.array([60, 200, 130])
    mask1 = cv2.inRange(yuv, a_low, a_high)

    a_low_x = np.array([185, 125, 120])
    a_high_x = np.array([210, 245, 140])
    mask2 = cv2.inRange(yuv, a_low_x, a_high_x)

    mask_a = cv2.bitwise_or(mask1, mask2)
    mask_a = cv2.bitwise_and(mask_a, mask_motion)

    b_low = np.array([80, 165, 75])
    b_high = np.array([105, 185, 90])
    mask3 = cv2.inRange(yuv, b_low, b_high)

    b_low_y = np.array([30, 110, 155])
    b_high_y = np.array([60, 120, 170])
    mask4 = cv2.inRange(yuv, b_low_y, b_high_y)

    mask_b = cv2.bitwise_or(mask3, mask4)
    mask_b = cv2.bitwise_and(mask_b, mask_motion)
    mask_b[0 : 360, 1000: ] = 0
    mask_b[400: , 0: ] = 0

    contours_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_a:
        area = cv2.contourArea(contour)
        if 600 < area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for contour in contours_b:
        area = cv2.contourArea(contour)
        if 60 < area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Frame4', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()
