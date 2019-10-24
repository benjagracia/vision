import cv2
import numpy as np
import math

lowerBound=np.array([20,77,67])
upperBound=np.array([190,231,215])

cam= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
xn = 0.112 #metros
yn = 0.088
zn = 0.23
alturaCam = 0.20

while True:
    ret, img=cam.read()
    #img=cv2.resize(img,(340,220))
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
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    num = 0
    for cont in conts:
        x,y,w,h=cv2.boundingRect(cont)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        
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
            
        xi1 = xi * k1      
        yi1 = yi * k1
        zi1 = zi * k1    
        
        #Ball in the floor approximation
        if (yi != 0):
            k2 = -alturaCam/yi
        else:
            k2 = 999999
        xi2 = xi * k2
        yi2 = yi * k2
        zi2 = zi * k2
        
        
        error = abs((k2 - k1)/k2)
        if error < 0.05 :
            num += 1
            xi = (xi1 + xi2)/2
            yi = (yi1 + yi2)/2
            zi = (zi1 + zi2)/2 
            print("ball "+str(num)+": "+ str(xi) + ", " + str(yi) + ", "+ str(zi))
            
            dispx = round(xi*1000)/1000
            dispy = round(yi*1000)/1000
            dispz = round(zi*1000)/1000
            dataStr = str(num)+"( "+str(dispx)+", "+str(dispy)+", "+str(dispz)+")"
            cv2.putText(img, dataStr, (x, y+10), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
            #cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
            #print(x,y)
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)
    