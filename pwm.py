import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# Motor
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)  # IN1
GPIO.setup(17, GPIO.OUT, initial=GPIO.LOW)  # IN2
GPIO.setup(23, GPIO.OUT, initial=GPIO.LOW)  # IN3
GPIO.setup(22, GPIO.OUT, initial=GPIO.LOW)  # IN4
#PWM port
GPIO.setup(13, GPIO.OUT)
GPIO.setup(12, GPIO.OUT, initial=GPIO.LOW)
#direction
GPIO.output(18, GPIO.HIGH)  # M1 in1
GPIO.output(17, GPIO.HIGH)  # M1 in2

GPIO.output(23, GPIO.HIGH)  # M2 in3
GPIO.output(22, GPIO.HIGH)  # M2 in4

#pwm jetson ports = gpio12, =gpio13
M1=GPIO.PWM(13, 50)
#M2=GPIO.PWM(12, 50)
#M2.start(0)
M1.start(0)
while True:
    r=input()
    if(r=="s"):
        M1.ChangeDutyCycle(10)
 #       M2.ChangeDutyCycle(10)
        print("slow")
    if(r=="m"):
        M1.ChangeDutyCycle(30)
  #      M2.ChangeDutyCycle(25)
        print("medium")
    if(r=="f"):
        print("fast")
        M1.ChangeDutyCycle(40)
   #     M2.ChangeDutyCycle(100)
    if(r=="h"):
        print("halt program")
        GPIO.output(18, GPIO.LOW)  # M1 in1
        GPIO.output(17, GPIO.LOW)  # M1 in2
        M1.stop()
    #    M2.stop()
        GPIO.cleanup()
        break
