import cv2
import numpy as np

RED = (0, 0, 255)
BLUE = (255, 0, 0)


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
    print("(width, height): ", (width, height))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (image.shape[1], image.shape[0]))

    # linspaceの引数はstart, stop, num(要素数)。
    for angle in np.linspace(0, 360, duration * fps):
        # print(angle)

        # アフィン変換による回転画像の作成
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255)) # 白埋め
        
        
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        # 二値化
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 白黒反転
        thresh = cv2.bitwise_not(thresh)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contours = max(contours, key=cv2.contourArea)
        # 画像中の全輪郭を描画
        # cv2.drawContours(rotated, contours, -1, RED, 3)
        
        # 0番目の輪郭のみを描画
        # cv2.drawContours(rotated, contours, 0, RED, 3)と同じ
        cnt = contours[0]
        cv2.drawContours(rotated, max_contours, 0, RED, 3)
        
        x, y, w, h = cv2.boundingRect(max_contours)
        cv2.rectangle(rotated, (x, y), (x+w, y+h), BLUE, 3)
        
        # 表示と書き込み
        video_writer.write(rotated)
        cv2.imshow("hehe", thresh)
        cv2.waitKey(1)
        
        
    cv2.destroyAllWindows()
    video_writer.release()


def main():
    image_path = './data/seat1.png'
    output_path = './data/box_with_max_contours_0324.mp4'
    duration = 10  # 動画の長さ（秒）
    fps = 30  # フレームレート

    # img = cv2.imread(image_path)
    # # cv2.line()
    # cv2.line(img, (20, 20), (200, 200), RED, 4)

    # cv2.imshow("hehe", img)
    # cv2.waitKey(0)


    generate_video(image_path, output_path, duration, fps)


if __name__ == "__main__":
    main()
