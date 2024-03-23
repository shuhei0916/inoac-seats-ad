import cv2
import numpy as np

RED = (0, 0, 255)

def find_largest_contour(image):
    """画像から最大の輪郭を見つける関数"""
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def calculate_centroid(contour):
    """輪郭の重心を計算する関数"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy


def draw_contour(image, contour, color):
    """画像に輪郭を描画する関数"""
    cv2.drawContours(image, [contour], -1, color, 2)


def draw_centroid(image, centroid, color):
    """画像に重心を描画する関数"""
    cv2.line(image, (centroid[0] - 20, centroid[1] - 20), (centroid[0] + 20, centroid[1] + 20), color, 2)
    cv2.line(image, (centroid[0] + 20, centroid[1] - 20), (centroid[0] - 20, centroid[1] + 20), color, 2)



def generate_video(image_path, output_path, duration, fps):
    """重心を中心に回転する動画を生成する関数"""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (image.shape[1], image.shape[0]))

    # linspaceの引数はstart, stop, num(要素数)。
    for angle in np.linspace(0, 360, duration * fps):
        # print(angle)

        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二値化
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 白黒反転
        thresh = cv2.bitwise_not(thresh)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # print(type(contours))
        # print(len(contours))
        # print(contours[0])
        
        
        # アフィン変換による回転画像の作成
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated = cv2.warpAffine(thresh, matrix, (width, height))
        
        # contoursがあんまよく分かってないので要調査
        # 最大の輪郭を取得
        cnt = max(contours, key=cv2.contourArea)

        # 最大の輪郭のみを描画 # 描画がうまくいっていない問題を調査する
        cv2.drawContours(rotated, [cnt], -1, RED, 2)
            
        
        # 表示と書き込み
        video_writer.write(rotated)
        cv2.imshow("hehe", rotated)
        cv2.waitKey(1)
        
        

    video_writer.release()


def main():
    image_path = './data/seat1.png'
    output_path = './data/contour_0324.mp4'
    duration = 10  # 動画の長さ（秒）
    fps = 30  # フレームレート


    generate_video(image_path, output_path, duration, fps)


if __name__ == "__main__":
    main()
