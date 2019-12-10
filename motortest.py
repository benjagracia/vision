import time
import Jetson.GPIO as GPIO
# MOTORS SETUP
# GPIO.cleanup() #clean trash from ports
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# Motor left
GPIO.setup(18, GPIO.OUT)  # IN1
GPIO.setup(17, GPIO.OUT)  # IN2
# Motor right
GPIO.setup(23, GPIO.OUT)  # IN3
GPIO.setup(22, GPIO.OUT)  # IN4

print('el motor se mueve')
GPIO.output(18, GPIO.HIGH)  # M1 in1
GPIO.output(17, GPIO.HIGH)  # M1 in2
GPIO.output(23, GPIO.HIGH)  # M2 in3
GPIO.output(22, GPIO.HIGH)  # M2 in4
time.sleep(10)
GPIO.output(18, GPIO.LOW)  # M1 in1
GPIO.output(17, GPIO.LOW)  # M1 in2
GPIO.output(23, GPIO.LOW)  # M2 in3
GPIO.output(22, GPIO.LOW)  # M2 in4
print('el motor dejo de moverse')

GPIO.cleanup() #clean trash from ports
