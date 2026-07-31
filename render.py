import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import imutils

img = cv2.imread ('image.png')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny (bfilter, 30,200)

keypoints= cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos =  imutils.grab_contours(keypoints)
contornos = sorted (contornos, key= cv2.contourArea, reverse= True)[0:10]


location = None
for contorno in contornos:
    aprox = cv2.approxPolyDP(contorno, 10, True)
    if len (aprox) == 4:
        location = aprox
        break



mascara = np.zeros (gray.shape, np.uint8)
nova_img= cv2.drawContours(mascara, [location], 0, 255, -1)
nova_img= cv2.bitwise_and(img, img, mask=mascara)




(x, y) = np.where(mascara == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

### plt.figure(figsize=(8,4))
#plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#plt.axis("off")
#plt.show()



leitor = easyocr.Reader(['pt'])
resultado = leitor.readtext(cropped_image)

print (resultado)