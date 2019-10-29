import cv2
import numpy as np
import imutils
import math
import time
import RPi.GPIO as GPIO
#MOTORES SETUP
#GPIO.cleanup()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT) #IN1
GPIO.setup(17,GPIO.OUT) #IN2S

GPIO.setup(23,GPIO.OUT) #IN3
GPIO.setup(22,GPIO.OUT) #IN4

lowerBound=np.array([29,85,6])
upperBound=np.array([64,255,255])

cam= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

xn = 0.112 #metros
yn = 0.088
zn = 0.23
alturaCam = 0.20

xl=list()
yl=list()
rl=list()

def average(dispx,dispy,radius,s):
        m=10
        xl.append(dispx)
        yl.append(dispy)
        rl.append(radius)
        if(len(xl)==m and len(yl)==m and len(rl)==m):
            xm=sum(xl)/len(xl)
            ym=sum(yl)/len(yl)
            rm=sum(rl)/len(rl)
            del yl[0:m]
            del xl[0:m]
            del rl[0:m]
            return (print("ball# " + str(s) + " x= " + str(round(xm, 4)) + " y= "+ str(round(ym, 4)) + " radius= ",str(int(round(rm)))))
        return

#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)

while True:
    ret, img=cam.read()
    img=cv2.resize(img,(600,500))
    
    xr = img.shape[1]
    yr = img.shape[0]
    blurred=cv2.GaussianBlur(img, (11,11),0)
    
    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    cnts = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    s=0
    for i in range(len(conts)):
        s+=1
        #((x,y),radius)=cv2.minEnclosingCircle(conts[i])
        #round(x)
        #round(y)
        #int(x,y)
        #int(radius)
        x,y,w,h=cv2.boundingRect(conts[i])
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        #cords= "x="+str(x)+" " + "y="+str(y)
        #cv2.putText(img, cords, (x+10, y+10), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        #cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
        c = max(cnts, key=cv2.contourArea)
        ((x2, y2), radius) = cv2.minEnclosingCircle(conts[i])
        #M = cv2.moments(c)
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        cv2.circle(img, (int(x2), int(y2)), int(radius),(0, 255, 255), 2)
        #cv2.circle(img, center, 5, (0, 0, 255), -1)
        
        #math
        r = math.sqrt(w*h)/2  #radio
        pry = (h/2)*2*yn/yr    #radio en metros en el plano
        xp = x+w/2             #centro en equis
        yp = y+h/2              #centro en y
        xi = xn*(2*xp/xr -1)    #vector al plano en x
        yi = yn*(1-2*yp/yr)      #vector al plano en y
        zi = zn                   #constante de z al plano
        m = math.sqrt(math.pow(xi,2)+math.pow(yi,2)+math.pow(zi,2))
        xi = xi/m            #vector direccion en x (sin mag)
        yi = yi/m            #vector direccion en y
        zi = zi/m            #vector direccion en z
        
        #Diameter approximation
        if (pry*zi != 0):
            k1 = zn * 0.033 / (pry* zi)   #.33=r de pelota de tennis , calculo de distancia a la pelota por medio del diametro
        else:
            k1 = 0
            
        xi = xi * k1      
        yi = yi * k1
        zi = zi * k1  
        
        #print("ball "+str(num)+": "+ str(xi) + ", " + str(yi) + ", "+ str(zi))
        
        dispx = round(xi*1000)/1000
        dispy = round(yi*1000)/1000
        dispz = round(zi*1000)/1000
        
        
        
        dataStr = str(s)+"( "+str(dispx)+", "+str(dispy)+", "+str(dispz)+")"
        cv2.putText(img, dataStr, (x, y), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        
        average(dispx,dispy,radius,s)
        #print("ball# "+ str(s) + " x= "+ str(dispx) + " y= " + str(dispy) +
        #      " radius= " + str(int(round(radius))))
        
        #print("ball# "+ str(s) + " x= "+ str(int(round(xi))) + " y= " + str(int(round(yi))) +
           #   " radius= " + str(int(round(radius))))
        #print("ball# "+ str(len(conts)-1) + " x= "+ str(x) + " y= " + str(y))
        
        if(k1<=2):
            if(dispx > 0):
                GPIO.output(18, GPIO.HIGH) #M1 in1
                GPIO.output(17, GPIO.LOW)  #M1 in2
                GPIO.output(23, GPIO.LOW)  #M2 in3
                GPIO.output(22, GPIO.HIGH) #M2 in4
                time.sleep(.05)
                GPIO.output(18, GPIO.LOW)  
                GPIO.output(17, GPIO.LOW)
                GPIO.output(23, GPIO.LOW)
                GPIO.output(22, GPIO.LOW)
                if(dispx<0.3):
                    GPIO.output(18, GPIO.HIGH) #M1 in1
                    GPIO.output(17, GPIO.LOW)  #M1 in2
                    GPIO.output(23, GPIO.LOW)  #M2 in3
                    GPIO.output(22, GPIO.HIGH) #M2 in4
                    time.sleep(.05)
                    GPIO.output(18, GPIO.LOW)  
                    GPIO.output(17, GPIO.LOW)
                    GPIO.output(23, GPIO.LOW)
                    GPIO.output(22, GPIO.LOW)
            if(dispx < 0):
                GPIO.output(18, GPIO.LOW)   #M1 in1
                GPIO.output(17, GPIO.HIGH)  #M1 in2
                GPIO.output(23, GPIO.HIGH)  #M2 in3
                GPIO.output(22, GPIO.LOW)   #M2 in4
                time.sleep(.05)
                GPIO.output(18, GPIO.LOW)  
                GPIO.output(17, GPIO.LOW)
                GPIO.output(23, GPIO.LOW)
                GPIO.output(22, GPIO.LOW)
                if(dispx>-0.3):
                    GPIO.output(18, GPIO.LOW)   #M1 in1
                    GPIO.output(17, GPIO.HIGH)  #M1 in2
                    GPIO.output(23, GPIO.HIGH)  #M2 in3
                    GPIO.output(22, GPIO.LOW)   #M2 in4
                    time.sleep(.02)
                    GPIO.output(18, GPIO.LOW)  
                    GPIO.output(17, GPIO.LOW)
                    GPIO.output(23, GPIO.LOW)
                    GPIO.output(22, GPIO.LOW)
            
            #GPIO.output(23, GPIO.HIGH) derecha
            #GPIO.output(24, GPIO.HIGH) izquierda
        
    
    
    cv2.imshow("maskClose",maskClose)
    #cv2.imshow("maskOpen",maskOpen)
    #cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)
    