import cv2
from keras.models import model_from_json
import numpy as np
import RPi.GPIO as GPIO
import time
import serial
import time
import pynmea2  # For parsing GPS data

# Set up GPIO pins for LEDs
green_led_pin = 17  # GPIO pin for green LED
red_led_pin = 18    # GPIO pin for red LED
GPIO.setmode(GPIO.BCM)
GPIO.setup(green_led_pin, GPIO.OUT)
GPIO.setup(red_led_pin, GPIO.OUT)

# Set up the software UART for the GSM module
GPIO.setmode(GPIO.BCM)
GSM_TX = 23  # GPIO pin used for GSM TX
GSM_RX = 24  # GPIO pin used for GSM RX

GPIO.setup(GSM_TX, GPIO.OUT)
GPIO.setup(GSM_RX, GPIO.IN)

gsm = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1)

# Set up the hardware UART for the GPS module
gps = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

# Load the pre-trained model
json_file = open("trainedbatchf2.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("trainedbatchf2.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 256, 256, 1)
    return feature / 255.0

# Function to send an SMS
def send_sms(number, message):
    gsm.write('AT+CMGF=1\r'.encode())
    time.sleep(1)
    gsm.write('AT+CMGS="{}"\r'.format(number).encode())
    time.sleep(1)
    gsm.write(message.encode())
    gsm.write(chr(26).encode())
    time.sleep(2)

# Function to get GPS coordinates
def get_gps_location():
    while True:
        data = gps.readline().decode('utf-8')
        if data.startswith('$GPGGA'):
            msg = pynmea2.parse(data)
            latitude = msg.latitude
            longitude = msg.longitude
            return f"Latitude: {latitude}, Longitude: {longitude}"


webcam = cv2.VideoCapture(0)
labels = {0: 'Drunk', 1: 'Sober'}
while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture a frame from the webcam.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (256, 256))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            
            if prediction_label == 'Drunk':
                # Turn on the red LED and prevent ignition
                GPIO.output(red_led_pin, GPIO.HIGH)
                # You can also implement the GSM module to send an SMS
                recipient_number = "+254714958170"
                message = "Driver is Drunk. Their location is: " + get_gps_location()
                # gsm.send_sms("Driver is Drunk. Location: " + gps_sensor.get_location())
                # Send the SMS
                send_sms(recipient_number, message)

            else:
                # Turn on the green LED and allow ignition
                GPIO.output(green_led_pin, GPIO.HIGH)

        cv2.imshow("Output", im)
        if cv2.waitKey(27) == 27:
            break
    except cv2.error:
        pass

gsm.close()
gps.close()
webcam.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # Cleanup GPIO pins
