import cv2
import numpy as np
import math
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
# define color detection boundaries
lowerBound = np.array([29, 85, 6])
upperBound = np.array([64, 255, 255])
# camera setup
dispW = 640
dispH = 480
flip = 2

camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1980, height=1080, format=NV12, framerate=59/1 ! nvvidconv flip-method=' + str(
    flip) + ' ! video/x-raw, width=' + str(dispW) + ', height=' + str(
    dispH) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
# setting up kernel properties with matrices
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((21, 21))
# camera position constants
xn = 0.112  # metros
yn = 0.088
zn = 0.23
alturaCam = 0.20
# lists for average function
xl = list()
yl = list()
rl = list()


# average function
def average(dispx, dispy, s):
    m = 10  # length of array accepted
    # appending values to each list
    xl.append(dispx)
    yl.append(dispy)
    # once the length of the generated lists equals m enter
    if len(xl) == m and len(yl) == m:
        # calculating the average measurements
        xm = sum(xl) / len(xl)
        ym = sum(yl) / len(yl)
        # cleaning up the lists
        del yl[0:m]
        del xl[0:m]
        # print results
        print("ball# " + str(s) + " x= " + str(round(xm, 4)) + " y= " + str(round(ym, 4)))
        return
    return

def cleanup():
    GPIO.output(18, GPIO.LOW)  # M1 in1
    GPIO.output(17, GPIO.LOW)  # M1 in2
    GPIO.output(23, GPIO.LOW)  # M2 in3
    GPIO.output(22, GPIO.LOW)
    GPIO.cleanup()

#(mdegans) wrapping this so that interrupt can be caught (see bottom of file)
def main():
    while True:
        # camera setup
        ret, img = cam.read()
        # img = cv2.resize(img, (600, 500))  # resolution size
        # native camera resolution
        xr = img.shape[1]
        yr = img.shape[0]
        #(mdegans): it doesn't look like these two blurred images are used, so commenting them out
        #(mdegans): you probably want to edit the code below to fix it
        # first blur
        # median = cv2.medianBlur(img, 7)
        # second blur
        # blurred = cv2.GaussianBlur(img, (11, 11), 0)
        # convert BGR to HSV
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # create the first mask filter
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
        # morphology, the second filter
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        # morphology, the third filter
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
        maskFinal = maskClose
        # quantity of contours from the final mask filter
        im2, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        s = 0

        if (len(conts) == 0):
            GPIO.output(18, GPIO.LOW)  # M1 in1
            GPIO.output(17, GPIO.LOW)  # M1 in2
            GPIO.output(23, GPIO.LOW)  # M2 in3
            GPIO.output(22, GPIO.LOW)
            print('stop')
        for i in range(len(conts)):
            s += 1  # ball counter
            # a = 1  # amount of balls permitted to see
            # drawing perceived contours
            cv2.drawContours(img, conts, -1, (255, 0, 0), 3)
            # drawing the minimum possible rectangle
            x, y, w, h = cv2.boundingRect(conts[i])
            # proceed only if minimum ball size threshold is met
            if w > 21 and h > 21:
                # draw a rectangle in the img to show it to the user
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # math for determining irl distance
                r = math.sqrt(w * h) / 2  # radius
                pry = (h / 2) * 2 * yn / yr  # radius in metres in the plane
                xp = x + w / 2  # center in x
                yp = y + h / 2  # center in y
                xi = xn * (2 * xp / xr - 1)  # vector pointed at the plane in x
                yi = yn * (1 - 2 * yp / yr)  # vector pointed at the plane in y
                zi = zn  # constant of z pointed at the plane
                m = math.sqrt(math.pow(xi, 2) + math.pow(yi, 2) + math.pow(zi, 2))
                xi = xi / m  # vector direction en x (sin mag)
                yi = yi / m  # vector direction en y
                zi = zi / m  # vector direction en z
                # Diameter approximation
                if pry * zi != 0:
                    k1 = zn * 0.033 / (
                            pry * zi)  # .33=r of tennis ball, calculus of distance from the ball in function of the diameter
                else:
                    k1 = 0
                    xi = xi * k1
                    yi = yi * k1
                    zi = zi * k1
                # rounding up final distances
                dispx = round(xi * 1000) / 1000
                dispy = round(yi * 1000) / 1000
                dispz = round(zi * 1000) / 1000
                # (mdegans): commented out display code because we are running headless now
                # making a string that includes coordinates to reduce clutter
                # dataStr = str(s) + "( " + str(dispx) + ", " + str(dispy) + ", " + str(dispz) + ")"
                # draw the coordinates in img to show it to the user
                # cv2.putText(img, dataStr, (x, y), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
                # call average function to smooth out calculated coordinates
                average(dispx, dispy, s)
                # movement of the robot

                # if ball detected within a 2 metres range proceed
                if (k1 <= 2):
                    if (dispx <= .2 and dispx >= -.2):
                        # foward
                        GPIO.output(18, GPIO.HIGH)  # M1 in1
                        GPIO.output(17, GPIO.LOW)  # M1 in2
                        GPIO.output(23, GPIO.HIGH)  # M2 in3
                        GPIO.output(22, GPIO.LOW)  # M2 in4
                        # time.sleep(.1)
                        print('foward')

                    # if the ball is in x positive, rotate to the right
                    if (dispx > .2):
                        GPIO.output(18, GPIO.LOW)  # M1 in1
                        GPIO.output(17, GPIO.HIGH)  # M1 in2
                        GPIO.output(23, GPIO.HIGH)  # M2 in3
                        GPIO.output(22, GPIO.LOW)  # M2 in4
                        print('right')
                    # if the ball is in x negative, rotate to the left
                    if (dispx < -.2):
                        GPIO.output(18, GPIO.HIGH)  # M1 in1
                        GPIO.output(17, GPIO.LOW)  # M1 in2
                        GPIO.output(23, GPIO.LOW)  # M2 in3
                        GPIO.output(22, GPIO.HIGH)  # M2 in4
                        print('left')
                        # if the ball is in y positive but closer to the center, rotate slower to the right
        # (mdegans): this part probably needs X to run and quitting is handled
        # below
        #     if cv2.waitKey(1) == ord('q'):
        #         cleanup()
        #         break
        # if cv2.waitKey(1) == ord('q'):
        #     cleanup()
        #     break
        # cv2.imshow("maskClose", maskClose)
        # # cv2.imshow("maskOpen",maskOpen)
        # # cv2.imshow("mask",mask)
        #cv2.imshow("cam", img)
        # cv2.waitKey(10)

if __name__ == '__main__':
    # (mdegans): I have wrapped your main() function in a try block to catch a KeyboardInterrupt (ctrl+c) or SIGINT
    # (mdegans): this ensures your script will close cleanly when you run "systemctl stop robot"
    try:
        main()
    except KeyboardInterrupt:
        print("Got SIGINT. Shutting down.")
        cleanup()
