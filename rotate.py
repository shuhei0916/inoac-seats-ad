import cv2
import numpy as np

def find_largest_contour_centroid(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    thresh = cv2.bitwise_not(thresh)
    
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


def rotate_image(image, centroid, angle):
    """画像を指定した角度分、重心を中心に回転させる関数"""
    height, width = image.shape[:2]

    # 回転行列の取得
    rotation_matrix = cv2.getRotationMatrix2D(centroid, angle, 1.0)  # scale=1.0で画像サイズ維持

    # アフィン変換で回転
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image



def generate_video(image, output_path, duration, fps):
    """重心を中心に回転する動画を生成する関数"""
    image, centroid = find_largest_contour_centroid(image)

    # 動画出力設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (image.shape[1], image.shape[0]))

    # 動画生成
    for angle in np.linspace(0, 360, duration * fps):
        rotated_image = rotate_image(image.copy(), centroid, angle)
  
        gray =  cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)
        
  
  
        video_writer.write(thresh)
        cv2.imshow('hehe', thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_writer.release()

def main():
    image_path = './data/seat1.png'
    output_path = './data/thresh_0323.mp4'
    duration = 10  # 動画の長さ（秒）
    fps = 30  # フレームレート

    img = cv2.imread(image_path)
    # resized = cv2.resize(img, None, fx=0.25, fy=0.25)
    
    # img = 
    # cv2.imshow("hehe", img)
    # cv2.waitKey(0)
    generate_video(image_path, output_path, duration, fps)


if __name__ == "__main__":
    main()
