#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
#검출기 불러오기
detector = MTCNN()

#이미지 불러오기
img_name = "ivan"
file_type = ".jpg"
image = cv2.imread(img_name + file_type)

#결과 result에 저장하기
result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
for x in range(len(result)):
    bounding_box = result[x]['box']
    keypoints = result[x]['keypoints']
    #bounding box 위치 그리기
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
    #5개의 얼굴 랜드마크 그리기
    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
cv2.imwrite(img_name + "_result" + file_type, image)
plt.imshow(image)
# cv2.imshow(img_name + "_result" + file_type, image)

print(result)