import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_folder = r'C:\Users\HP\Downloads\Glaucoma-Detection-master\Glaucoma-Detection-master\data\train\not_glaucoma'

def calculate_cdr(image_path, figure_number):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 2)
    blurred_norm = cv2.normalize(blurred, None, 0, 1, cv2.NORM_MINMAX)

    _, disc_thresh = cv2.threshold(blurred_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    disc_thresh = cv2.morphologyEx(disc_thresh.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    disc_thresh = cv2.morphologyEx(disc_thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(disc_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('No optic disc detected in image:', os.path.basename(image_path))
        return None

    disc_contour = max(contours, key=cv2.contourArea)
    disc_area = cv2.contourArea(disc_contour)

    x, y, w, h = cv2.boundingRect(disc_contour)
    disc_cropped = gray[y:y+h, x:x+w]
    disc_cropped_norm = cv2.normalize(disc_cropped, None, 0, 1, cv2.NORM_MINMAX)

    _, cup_binary = cv2.threshold(disc_cropped_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cup_binary = cv2.morphologyEx(cup_binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cup_binary = cv2.morphologyEx(cup_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cup_contours, _ = cv2.findContours(cup_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cup_contours:
        print('No optic cup detected in image:', os.path.basename(image_path))
        return None

    cup_contour = max(cup_contours, key=cv2.contourArea)
    cup_area = cv2.contourArea(cup_contour)

    cdr = cup_area / disc_area
    print('Image:', os.path.basename(image_path), 'Cup-to-Disc Ratio (CDR): {:.2f}'.format(cdr))

    glaucoma_status = 'Glaucoma Detected' if cdr > 0.5 else 'No Glaucoma Detected'

    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Figure {figure_number}', fontsize=16)  # Sequential figure number
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(disc_thresh, cmap='gray')
    plt.title('Optic Disc Detected')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cup_binary, cmap='gray')
    plt.title(f'Optic Cup Detected, CDR: {cdr:.2f}')

    plt.show()

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Result: {glaucoma_status}')
    plt.axis('off')
    plt.show()

# Initialize figure number
figure_number = 1

# Process all images in the dataset
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        calculate_cdr(os.path.join(image_folder, image_file), figure_number)
        figure_number += 1  # Increment the figure number
