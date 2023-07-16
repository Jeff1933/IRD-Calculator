import cv2
import streamlit as st
import numpy as np
def count_dark_matter_content(img):
    # Resize the image to fit the desired height
    #img = cv2.resize(img, (int(img.shape[1] * 1550 / img.shape[0]), 1550))
    
    # Define the region of interest
    roi = img[:, int(img.shape[1] / 6.5):]
    roi = roi[:, :-int(img.shape[1] / 7.5)]
    
    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(gray, 190, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 311, 7)
    
    # Find contours
    contours, hierarchy = cv2.findContours(adaptive_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    marked_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > 60000:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.007 * perimeter, True)
        if len(approx) >= 5 and len(approx) <= 30:
            marked_contours.append(contour)
            cv2.drawContours(roi, [contour], 0, (0, 255, 0), 2)
    
    target_pixels = 0
    for contour in marked_contours:
        target_pixels += cv2.contourArea(contour)
    
    target_ratio = target_pixels / (gray.shape[0] * gray.shape[1])
    return target_ratio

def calculate_target_ratio(img, slider_value):
    # 根据滑动栏的值确定有效识别区域
    roi = img[slider_value:slider_value+1550, :]
    
    # 计算并返回target_ratio
    return count_dark_matter_content(roi)

def main():
    st.title("Ice-Rafted Debris Calculator")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        
        # 添加滑动栏
        slider_value = st.slider('Select ROI position', 0, image.shape[0]-1550, 0)
        
        # 使用新的函数计算target_ratio
        target_ratio = calculate_target_ratio(image, slider_value)
        
        st.image(image[slider_value:slider_value+1550, :], caption="Uploaded Image", use_column_width=True)
        st.write("Target Ratio:", target_ratio)

if __name__ == "__main__":
    main()