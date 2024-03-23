import numpy as np
import cv2

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def main():
    
    
    img = cv2.imread('./data/seat1.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    
    img = cv2.drawContours(img, contours, -1, RED, 3)
    

    cv2.imshow('hehe', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()