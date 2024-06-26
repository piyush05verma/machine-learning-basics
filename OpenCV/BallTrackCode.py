import cv2
import numpy as np

cap = cv2.VideoCapture('OpenCV/volleyball_match.mp4')

detector = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blur = cv2.GaussianBlur(frame, (5, 5), sigmaX = 13)
    yuv = cv2.cvtColor(blur, cv2.COLOR_RGB2YUV)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_RGB2YCrCb)
    
    ball_low = np.array([80, 160, 45])
    ball_high = np.array([200, 180, 110])
    mask1 = cv2.inRange(yuv, ball_low, ball_high)

    ball_low = np.array([0, 100, 130])
    ball_high = np.array([20, 230, 255])
    mask2 = cv2.inRange(hsv, ball_low, ball_high)

    mask3 = cv2.bitwise_and(mask1, mask2)

    mask4 = detector.apply(blur)
    mask = cv2.bitwise_and(mask3, mask4)

    #kernel = np.ones((3,3),np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    '''
    cv2.RETR_EXTERNAL: This mode retrieves only the external contours, i.e., the contours that form the outer boundary of the objects in the image. Internal contours are not retrieved. This is useful when you are only interested in the overall shape of the objects.
    cv2.RETR_LIST: This mode retrieves all of the contours in the image but does not establish any hierarchical relationships between them. It simply returns a flat list of contours.
    cv2.RETR_CCOMP: This mode retrieves all of the contours and organizes them into a two-level hierarchy. The top-level contours represent the external boundaries of the objects, while the second-level contours represent the boundaries of any internal holes within the objects.
    cv2.RETR_TREE: This mode retrieves all of the contours and reconstructs a full hierarchy of nested contours. Each contour is linked to its parent, child, and sibling contours, allowing for more complex relationships between contours to be analyzed.
    
    cv2.CHAIN_APPROX_NONE: In this method, all the contour points are stored. No approximation is performed, so the number of points in the contour remains the same as the original contour. 
    cv2.CHAIN_APPROX_SIMPLE: This method compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, if a contour is a straight line, it only saves the two end points of that line.
    cv2.CHAIN_APPROX_TC89_L1 and cv2.CHAIN_APPROX_TC89_KCOS: They use a recursive algorithm to approximate the contour with fewer points while minimizing the maximum deviation between the original and simplified contour.
    
    '''

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if 100 < area < 300 and 35 < perimeter < 70:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) <= 10:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow('Frame4', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()