import cv2
import matplotlib.pyplot as plt


import pytesseract
import re
import numpy as np


img = cv2.imread('data/test4.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

center = img.shape[1] / 2

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

candidats = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if x <= center and x + w >= center and area > 1000:
        candidats.append(cnt)


def find_most_similar(data, n):
    data = sorted(data, key= lambda x: cv2.contourArea(x))
    min_diff = float('inf')
    best_start_idx = 0
    
    for i in range(len(data) - n + 1):
        current_diff = cv2.contourArea(data[i + n - 1]) - cv2.contourArea(data[i])
        
        if current_diff < min_diff:
            min_diff = current_diff
            best_start_idx = i
            
    return data[best_start_idx : best_start_idx + n]


clears = []

for cnt in find_most_similar(candidats, 4):
    x, y, w, h = cv2.boundingRect(cnt)

    # 1. Crop from the original BGR image
    cropped_bgr = img[y:y+h, x:x+w]

    # 2. Convert to Grayscale
    cropped_gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Refine with OpenCV Erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_img = cv2.erode(cropped_gray, kernel, iterations=1)

    # 4. Cleanup background noise 
    # Logic: Find most common color and fill dark/noisy spots
    values, counts = np.unique(refined_img, return_counts=True)
    common_color = values[np.argmax(counts)]
    
    result = refined_img.copy()
    # If the image is 0-255, we use a threshold like 127 instead of 0.5
    result[result < 127] = common_color

    # 5. Prepare for Tesseract (Black text on White background is best)
    # We ensure it's uint8. You might also want to apply a binary threshold here.
    _, roi = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if text is white on black; if so, invert it
    # Tesseract prefers black text on white background
    if np.mean(roi) < 127:
        roi = cv2.bitwise_not(roi)

    # 6. OCR
    text = pytesseract.image_to_string(roi, config='--psm 7')
    line = re.findall(r'[a-zA-Z0-9]+', text)
    res = []
    for part in line:
        if part.isdigit():
            res.append(int(part))
        elif len(part) > 1:
            res.append(part)
    
    if len(res) == 3:
        clears.append(res)

print(clears)