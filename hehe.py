import cv2
import numpy as np

def draw_centroid(image):
  """
  入力画像の輪郭を抽出し、その輪郭内の重心を描画する関数

  Args:
    image: 入力画像

  Returns:
    重心を描画した画像
  """

  # 画像をグレースケールに変換
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # 輪郭抽出
  contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # 各輪郭の重心を計算
  for contour in contours:
    # 輪郭のモーメントを計算
    moments = cv2.moments(contour)

    # 重心を計算
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    # 重心を画像に描画
    cv2.circle(image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

  return image

# 入力画像の読み込み
input_image = cv2.imread("./data/seat1.jpg")

# 重心を描画
output_image = draw_centroid(input_image)

# 出力画像の表示
output_image = cv2.resize(output_image, None, fx=0.25, fy=0.25)
cv2.imshow("Output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
