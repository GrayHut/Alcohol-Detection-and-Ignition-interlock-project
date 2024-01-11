import cv2
import numpy as np
import tflite_runtime.interpreter as interpreter
import serial
import time
import spidev

# Set up the software SPI for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Chip-select 0

# Set up the GSM module
gsm = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)

# Load the TFLite model
interpreter = interpreter.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

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
    # Adjust this conversion factor based on your sensor calibration
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

webcam = cv2.VideoCapture(0)
labels = {0: 'Drunk', 1: 'Sober'}


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


            highest_reading = 0

            for _ in range(5):
                mq3_channel = 0
                mq3_value = read_mq3_sensor(mq3_channel)
                highest_reading = max(highest_reading, mq3_value)
                print(f"Sensor Reading: {mq3_value}")
                time.sleep(2)

            if prediction_label == 'Drunk':
                phone_number = "+254722209026"
                highest_reading_grams_per_ml = convert_to_grams_per_ml(highest_reading)
                message_content = f"Highest MQ3 Sensor Reading: {highest_reading}, Grams per ml: {highest_reading_grams_per_ml}"
                send_sms(phone_number, message_content)

        except KeyboardInterrupt:
            print("Program terminated by user.")
        finally:
            # Close the SPI and GSM connections
            spi.close()
            gsm.close()

    # Display the captured frame
    cv2.imshow("Captured Frame", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
