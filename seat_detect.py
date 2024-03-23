# Import necessary libraries
import cv2
import numpy as np

# Re-define the function to include necessary imports
def find_largest_contour_centroid(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the first one
    max_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the largest contour
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Draw the largest contour
    cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)

    # Draw the centroid of the largest contour
    cv2.line(image, (cx-20, cy-20), (cx+20, cy+20), (0, 0, 255), 2)
    cv2.line(image, (cx+20, cy-20), (cx-20, cy+20), (0, 0, 255), 2)

    return image, (cx, cy)

def main():
    # Now let's apply the function to the uploaded image
    image_path = './data/seat1.jpg'
    edited_image, centroid = find_largest_contour_centroid(image_path)

    # Save the edited image
    edited_image_path = './data/res2_0322.jpg'
    cv2.imwrite(edited_image_path, edited_image)

    # Return the path to the edited image and the centroid coordinates
    edited_image_path, centroid


if __name__ == "__main__":
    main()