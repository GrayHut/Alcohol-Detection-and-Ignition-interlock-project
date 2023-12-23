import cv2
import numpy as np
import tflite_runtime.interpreter as interpreter
import serial
import time
import spidev
import RPi.GPIO as GPIO
import pynmea2

# Set up the software SPI for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Chip-select 0

# Set up the GSM module
gsm = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)

# Set up GPIO pins for LEDs
GPIO.setmode(GPIO.BCM)
RED_LED_PIN = 7
GREEN_LED_PIN = 6
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# Set up the GPS module
gps = serial.Serial('/dev/ttyS0', baudrate=9600, timeout=1)

# Load the TFLite model
model_path = "model.tflite"  # path to the TFLite model
interpreter = interpreter.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def read_mq3_sensor(channel):
    # MCP3008 configuration
    adc_command = 0b11000000 | ((channel & 0x07) << 3)

    # Perform SPI transaction and read analog value
    spi_data = spi.xfer2([1, adc_command, 0])
    adc_value = ((spi_data[1] & 0x03) << 8) + spi_data[2]
    return adc_value

def convert_to_grams_per_ml(sensor_value):
    # Function to convert sensor value to grams per ml
    # This conversion is specific to the MQ-3 sensor and may need calibration
    conversion_factor = 0.01
    grams_per_ml = sensor_value * conversion_factor
    return grams_per_ml

def send_sms(phone_number, message):
    try:
        # Set the SMS mode
        gsm.write(('AT+CMGF=1\r').encode())
        time.sleep(1)

        # Set the phone number
        gsm.write(('AT+CMGS="{}"\r'.format(phone_number)).encode())
        time.sleep(1)

        # Send the message
        gsm.write((message + '\x1A').encode())
        time.sleep(1)

        print("SMS sent successfully!")

    except Exception as e:
        print(f"Error: {e}")

def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 256, 256, 1)
    return image / 255.0

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

try:
    # Get user input for phone number
    phone_number = input("Enter the cellular phone number: ")

    while True:
        ret, im = webcam.read()
        if not ret:
            print("Failed to capture a frame from the webcam.")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)

        for (p, q, r, s) in faces:
            try:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (256, 256))
                img = extract_features(image)

                img = img.astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()

                pred = interpreter.get_tensor(output_details[0]['index'])
                prediction_label = labels[pred.argmax()]

                cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

                if prediction_label == 'Drunk':
                    # Turn on Red LED and off Green LED
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

                    highest_reading = 0

                    for _ in range(5):
                        mq3_channel = 0
                        mq3_value = read_mq3_sensor(mq3_channel)
                        highest_reading = max(highest_reading, mq3_value)
                        print(f"Sensor Reading: {mq3_value}")
                        time.sleep(2)

                    if highest_reading_grams_per_ml > 0.08:
                        gps_data = get_gps_location()
                        highest_reading_grams_per_ml = convert_to_grams_per_ml(highest_reading)
                        message_content = f"Highest MQ3 Sensor Reading: {highest_reading}, Grams per ml: {highest_reading_grams_per_ml}\nGPS Location: {gps_data}"
                        send_sms(phone_number, message_content)
                    else:
                        print("Breath sample not taken.")

                elif prediction_label == 'Sober':
                    # Turn on Green LED and off Red LED
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    GPIO.output(RED_LED_PIN, GPIO.LOW)

            except KeyboardInterrupt:
                print("Program terminated by user.")

        # Display the captured frame
        cv2.imshow("Captured Frame", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the SPI and GSM connections if they are open
    if spi and spi.open:
        spi.close()
    if gsm and gsm.is_open:
        gsm.close()
    # Cleanup GPIO
    GPIO.cleanup()

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()
