import numpy as np
import cv2
import shutil, os

def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    R, G, B = cv2.split(np.asarray(image))
    output1_R = clahe.apply(R)
    output1_G = clahe.apply(G)
    output1_B = clahe.apply(B)
    image = cv2.merge((output1_R, output1_G, output1_B))
    return image

def main():
    L = []
    for root, dirs, files in os.walk('D:/dataset/MIO-TCD-Localization/MIO-TCD-Localization/_480x720'): 
        L.append(files)
    for file in L[0]:
        img = cv2.imread('D:/dataset/MIO-TCD-Localization/MIO-TCD-Localization/_480x720/'+file)
        enhanced_image = CLAHE(img)
        cv2.imwrite('D:/dataset/clahe_MIO-TCD_720_480/'+file, enhanced_image)
        
if __name__ == '__main__':
    main()
