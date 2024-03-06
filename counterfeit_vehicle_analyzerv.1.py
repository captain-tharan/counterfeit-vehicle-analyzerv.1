import cv2
import numpy as np
import pytesseract
import sqlite3
import smtplib
from email.message import EmailMessage
import ssl
import time

checked_plates = set()

def process_frame(frame):
    global checked_plates
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray_blur, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    vehicle_roi = frame[y:y+h, x:x+w]
    n_colors = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    pixels = vehicle_roi.reshape(-1, 3).astype(np.float32)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    color_counts = np.bincount(labels.flatten())
    dominant_color = palette[np.argmax(color_counts)]
    dominant_color = tuple(reversed(dominant_color))
    plate_roi = gray[y:y+h, x:x+w]
    thresh_plate = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    plate_texts = pytesseract.image_to_string(thresh_plate, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    plate_text = plate_texts.rstrip()
    if w/h > 1.5 and w/h <= 2.5:
        vehicle_type = "car"
    elif w/h > 2.5 and w/h <= 4.0:
        vehicle_type = "truck"
    elif w/h < 1.5:
        vehicle_type = "bike"
    elif w/h > 4.0 and w/h <= 5.0:
        vehicle_type = "SUV"
    else:
        vehicle_type = "unknown"

    print("Vehicle type: ", vehicle_type)
    if dominant_color[0] > dominant_color[1] and dominant_color[0] > dominant_color[2]:
        dominant_color="red"
    elif dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]:
        dominant_color="blue"
    elif dominant_color[2] > dominant_color[0] and dominant_color[2] > dominant_color[1]:
        dominant_color="green"
    elif dominant_color[0] == dominant_color[1] and dominant_color[0] == dominant_color[2]:
        dominant_color="gray"
    elif dominant_color[0] == dominant_color[1]:
        dominant_color="yellow"
    elif dominant_color[0] == dominant_color[2]:
        dominant_color="purple"
    elif dominant_color[1] == dominant_color[2]:
        dominant_color="teal"
    else:
        dominant_color="unknown"
    print("Vehicle color: ", dominant_color)
    print("License plate number: ", plate_text)

    if plate_text in checked_plates:
        print("This vehicle has already been checked.")
        return
    else:
        checked_plates.add(plate_text)

    # Connect to the SQLite database
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    c.execute('SELECT * FROM vehicles WHERE license_plate = ?', (plate_text,))
    result = c.fetchone()
    if result is not None:
        if result[1] == vehicle_type and result[2] == str(dominant_color):
            print("This is an original vehicle.")
        else:
            print("This is a counterfeit vehicle.")
            email_sender="sender_gmail@gmail.com"
            email_password=''
            email_reciver="reciver_gmail@gmail.com"
            subject="Fake vehicle detected"
            body=("The vehicle with license plate {} is fake!".format(plate_text))
            em=EmailMessage()
            em['From']=email_sender
            em['To']=email_reciver
            em['subject']=subject
            em.set_content(body)

            context=ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
                smtp.login(email_sender,email_password)
                smtp.sendmail(email_sender,email_reciver,em.as_string())
    else:
        print("This is a new vehicle.")
    conn.close()

def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    time_interval = 2.0

    while video_capture.isOpened():
        start_time = time.time()
        ret, frame = video_capture.read()

        if not ret:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(max(0, time_interval - (time.time() - start_time)))

    video_capture.release()
    cv2.destroyAllWindows()

video_path = "input_vedio_file_for_check.mp4"
process_video(video_path)
