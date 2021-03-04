# src: https://stackoverflow.com/questions/54082300/how-to-create-a-transparent-mask-in-opencv-python
import cv2
import numpy as np
from os import walk

CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100
input_folder = "images/"
output_folder = "output/"

def cut(img):
  # crop image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #-- Edge detection
  edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
  kernel = np.ones((7,7),np.uint8)
  edges = cv2.dilate(edges, kernel, iterations = 2)
  #edges = cv2.erode(edges, kernel)


  cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  cnt = sorted(cnts, key=cv2.contourArea)[-1]
  x,y,w,h = cv2.boundingRect(cnt)
  new_img = img[y:y+h, x:x+w]

  return new_img

def transBg(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #-- Edge detection
  edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
  kernel = np.ones((7,7),np.uint8)
  edges = cv2.dilate(edges, kernel, iterations = 2)
  #edges = cv2.erode(edges, kernel)


  roi, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  cnt = sorted(roi, key=cv2.contourArea)[-1]

  mask = np.zeros(img.shape, img.dtype)

  cv2.fillPoly(mask, [cnt], (255,)*img.shape[2], )

  masked_image = cv2.bitwise_and(img, mask)

  return masked_image

def fourChannels(img):
  height, width, channels = img.shape
  if channels < 4:
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return new_img

  return img

def main():
  _, _, filenames = next(walk(input_folder))

  print(filenames)
  for file_name in filenames:
      s_img = cv2.imread(input_folder + file_name, -1)

      # set to 4 channels
      s_img = fourChannels(s_img)

      # set background transparent
      s_img = transBg(s_img)

      # remove white background
      s_img = cut(s_img)

      cv2.imwrite(output_folder + file_name, s_img)

      s_img = None

main()
