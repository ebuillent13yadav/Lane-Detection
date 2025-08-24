import cv2
import numpy as np 

def regionOfInterest(img):
    # drawing trapezoid
    height,width = img.shape[:2]
    
    polygons = np.array([[(100,height),(img.shape[1]-100 ,height),
                          (int(width * 0.6),height//2),(int(width * 0.4),height//2)]])
    
    mask = np.zeros_like(img)
    
    cv2.fillPoly(mask,polygons,255)
    return cv2.bitwise_and(img, mask)


def drawLines(img , lines):
    # drawing lines on the image
    lines_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_img, (x1,y1), (x2,y2), (0,255,0), 5)
    return cv2.addWeighted(img,0.8,lines_img,1.0,0)

# Reading video
capture = cv2.VideoCapture("test_video.mp4")

while capture.isOpened():
    isTrue, frame = capture.read()
    if not isTrue :
        break
    
    # Pre processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur, 50, 150)
    
    #ROI
    cropped_edges = regionOfInterest(edge)
    
    # Hough Line Transformation
    lines = cv2.HoughLinesP(cropped_edges,2,np.pi/180,50,np.array([]), 40 ,100)
    
    # Drawing lines
    output = drawLines(frame,lines)
    
    # showing output
    
    cv2.imshow("lane detection", output)
    
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break

capture.release()
cv2.destroyAllWindows()

