import cv2
import numpy as np

# 画像を読み込む
image = cv2.imread('./data/seat1.jpg')  # 画像のパスを指定

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 閾値処理を行い、二値画像を作成
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 輪郭を検出
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 元の画像に輪郭と重心を描画
for contour in contours:
    # 輪郭の重心を計算
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # 重心に小さな円を描画
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    # 輪郭を描画
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# 結果を表示
image = cv2.resize(image, None, fx=0.25, fy=0.25)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
