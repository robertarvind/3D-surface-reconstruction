import imutils
import cv2
import math
import numpy as np
import fire
from numpy import array
from pylab import *
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import imread
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
from scipy import interpolate
import mpl_toolkits.mplot3d.axes3d as axes3d
from bisect import bisect_left
from bisect import bisect_right
from statistics import median
from numpy import median
from numpy import zeros
from numpy import empty
from operator import itemgetter

zz = []
znz = []
dzz = []
#ezz = []
#r1r = []
#s1s = []
#xx1 = []
#yy1 = []
#xx2 = []
#yy2 = []
#xx3 = []
#yy3 = []
#xx4 = []
#yy4 = []
dists = []
dists2 = []
#new = []
#neww = []
#edisty = []
#distys = []
#ndists = []
#rr = []
#ss = []
#tt = []
#uu = []
#rrx = []
#ssx = []
#ttx = []
#uux = []
#znccl = []
jptl = []
jjl = []
msuml = []
r1msuml = []
r2msuml = []
msuml1 = []
msuml2 = []
rmsuml = []
rxmsuml = []
rymsuml = []
rzmsuml = []
ar1msuml = []
ar2msuml = []
ar3msuml = []
msumll1 = []
msumll2 = []
msumls = []
vnml = []
vnmls = []
indexx = []
imcrpl1 = []
imcrpl2 = []
srdistl = []
srl = []
rzim111 = []
rzim222 = []
r1zim111 = []
r1zim222 = []
r2zim111 = []
r2zim222 = []
r3zim111 = []
r3zim222 = []
r4zim111 = []
r4zim222 = []
maxlist = []
#jlo = []
#jlo1 = []
#arr = [[0 for rrow in range(rowws)] for ccol in range(0,5)]
#ii = []
#imcrpl1 = []
#imcrpl2 = []
#op = []

img1 = cv2.imread('Dubey Rt 15cr.png',0)
#img1 = cv2.GaussianBlur(img1, (5, 5), 0)
rows1, cols1 = img1.shape
print(rows1)
print(cols1)
#cclahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#ccl1 = cclahe1.apply(img1)
img2 = cv2.imread('Dubey Rt 20cr.png',0)
#img2 = cv2.GaussianBlur(img2, (5, 5), 0)
rows2, cols2 = img2.shape
#cclahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#ccl2 = cclahe2.apply(img2)
iimmgg = cv2.imread('Dubey Rt 15cr1.png',0)
rowws, colls = iimmgg.shape
#gray1 = cv2.GaussianBlur(img1, (41, 41), 0)
#(minVal1, maxVal1, minLoc1, maxLoc1) = cv2.minMaxLoc(gray1)
imgg1 = img1#[maxLoc1[1]-150:maxLoc1[1]+150, maxLoc1[0]-150:maxLoc1[0]+150]
#imggg1 = cv2.GaussianBlur(imgg1, (41, 41), 0)
#cv2.imshow('imgg1',imgg1)
#plt.hist(imgg1.ravel(),256,[0,256]); plt.show()

arr = [[0 for rrow in range(rows2)] for ccol in range(cols2)]
#arr = [[]]
#arx, ary = arr.shape

#gray2 = cv2.GaussianBlur(img2, (41, 41), 0)
#(minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(gray2)
imgg2 = img2#[maxLoc2[1]-150:maxLoc2[1]+150, maxLoc2[0]-150:maxLoc2[0]+150]
#imggg2 = cv2.GaussianBlur(imgg2, (41, 41), 0)
#cv2.imshow('imgg2',imgg2)
#plt.hist(imgg2.ravel(),256,[0,256]); plt.show()
#img3 = cv2.imread('housestereo1.tif',0)
#rows3, cols3 = img3.shape
#imgg3 = img3
#img4 = cv2.imread('housestereo2.tif',0)
#rows4, cols4 = img4.shape
#imgg4 = img4
#clahe1 = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(128,128))
#cl1 = clahe1.apply(imgg1)
#equ1 = cv2.equalizeHist(imgg1)
#ccl1 = cv2.GaussianBlur(cl1, (41, 41), 0)
#crows1, ccols1 = cl1.shape
#xv1, yv1 = np.meshgrid(range(cols1), range(rows1)[::-1])
#cv2.imshow('cl1',cl1)
#plt.hist(cl1.ravel(),256,[0,256]); plt.show()

#clahe2 = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(128,128))
#cl2 = clahe2.apply(imgg2)
#equ2 = cv2.equalizeHist(imgg2)
#ccl2 = cv2.GaussianBlur(cl2, (41, 41), 0)
#crows2, ccols2 = cl2.shape
#xv2, yv2 = np.meshgrid(range(cols2), range(rows2)[::-1])
#cv2.imshow('cl2',cl2)
#plt.hist(cl2.ravel(),256,[0,256]); plt.show()
#sz = 10
#szz = ((sz-1)**2) + 1
n = 0
et = 2
vw = 200
bm = 6
hf = 31
#jjls = set(jjl)
for x1 in range(vw,rows1-vw):
 #x1 = x1 + 1
 #rr.append(x1)
 
 for y1 in range(vw,cols1-vw):
 
  #y1 = y1 + 1
  #p = int(img1[x1,y1])
  #for xp2 in range(2,rows2-2):
   #for yp2 in range(2,cols2-2):
    #srdist = math.sqrt(((xp2 - x1)**2)+((yp2 - y1)**2))
    #if srdist <= sz*(math.sqrt(2)):
     #srdistl.append(srdist)
     #srdistl.sort()

  #srdistl.sort()
  #srdistlol = srdistl[szz]
  #print(len(srdistl))
  #print(srdistlol)
  #del srdistl[:]
  #if x1 >= 1 and x1 <= (rows1 - 1) and y1 >= 1 and y1 <= (cols1 - 1):
  #p = img1[x1,y1]
  #p1 = img1[x1-1,y1-1]
  #p2 = img1[x1-1,y1]
  #p3 = img1[x1-1,y1+1]
  #p4 = img1[x1,y1-1]
  #p5 = img1[x1,y1+1]
  #p6 = img1[x1+1,y1-1]
  #p7 = img1[x1+1,y1]
  #p8 = img1[x1+1,y1+1]
  #p9 = img1[x1+2,y1+2]
  #p10 = img1[x1-2,y1-2]
  #p11 = img1[x1+1,y1+2]
  #p12 = img1[x1+2,y1+1]
  #p13 = img1[x1,y1+2]
  #p14 = img1[x1+2,y1]
  #p15 = img1[x1-1,y1+2]
  #p16 = img1[x1+2,y1-1]
  #p17 = img1[x1+2,y1-2]
  #p18 = img1[x1-2,y1+2]
  #p19 = img1[x1-2,y1+1]
  #p20 = img1[x1+1,y1-2]
  #p21 = img1[x1,y1-2]
  #p22 = img1[x1-2,y1]
  #p23 = img1[x1-1,y1-2]
  #p24 = img1[x1-2,y1-1]
  #mp = p + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
  #mpp = mp + p9 + p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 + p19 + p20 + p21 + p22 + p23 + p24
  #mp = mp/9
  #mpp = mpp/25
  #sdm1 = float((((p-mp)**2)+((p1-mp)**2)+((p2-mp)**2)+((p3-mp)**2)+((p4-mp)**2)+((p5-mp)**2)+((p6-mp)**2)+((p7-mp)**2)+((p8-mp)**2))/8)
  #imcr1 = imgg1[y1-(et-2):y1+(et-2), x1-(et-2):x1+(et-2)]
  imcr11 = imgg1[y1-(et-1):y1+(et), x1-(et-1):x1+(et)]
  imcr111 = imgg1[y1-et:y1+(et+1), x1-et:x1+(et+1)]
  #print(x1)
  #print(y1)
  #print(imgg1[x1,y1])
  #rimcr = 
  #imcr11 = cv2.GaussianBlur(imcrr1, (3, 3), 0)

  #imccr1 = imcrr1 - imcr11
  #ret,th = cv2.threshold(imccr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  #median = cv2.medianBlur(th,3)
   #median1 = cv2.medianBlur(median,5)
   #median2 = cv2.medianBlur(median1,5)
   #median3 = cv2.medianBlur(median2,5)
   #median4 = cv2.medianBlur(median3,5)
  #f = np.fft.fft2(median)
  #fshift = np.fft.fftshift(f)
  #imcr1 = 20*np.log(np.abs(fshift)) #imccr1
  #rws1, cls1 = imcr1.shape
  rws11, cls11 = imcr11.shape
  rws111, cls111 = imcr111.shape
  #print(imcr111)
  #print(rws111)
  #print(cls11)
  #rzimcr111 = cv2.resize(imcr111,(rws111-2, cls111-2), interpolation = cv2.INTER_CUBIC) #cv2.INTER_CUBIC
  #rxzimcr111 = cv2.resize(imcr111,(rws111-2, cls111-2), interpolation = cv2.INTER_LINEAR) #cv2.INTER_LINEAR
  #ryzimcr111 = cv2.resize(imcr111,(rws111-2, cls111-2), interpolation = cv2.INTER_AREA)   #cv2.INTER_AREA
  #rzzimcr111 = cv2.resize(imcr111,(rws111-2, cls111-2), interpolation = cv2.INTER_NEAREST) #cv2.INTER_NEAREST
  #io = 0
  #rzrw111, rzcl111 = rzimcr111.shape
  #rzrw111 = rws111 - 1
  #rzcl111 = cls111 - 1
  
  #avgsum1 = 0.0
  #for a1 in range(0,rws1):
   #for b1 in range(0,cls1):
    #impix1 = imcr1[a1,b1]
    #avgsum1 = avgsum1 + impix1

  #mp = avgsum1/(rws1*cls1)

  avgsum11 = 0.0
  for a11 in range(0,rws11):
   for b11 in range(0,cls11):
    impix11 = imcr11[a11,b11]
    avgsum11 = avgsum11 + impix11

  mpp = avgsum11/(rws11*cls11)

  #c = 0

  avgsum111 = 0.0
  for a111 in range(0,rws111):
   for b111 in range(0,cls111):
    impix111 = imcr111[a111,b111]
    avgsum111 = avgsum111 + impix111

  mppp = avgsum111/(rws111*cls111)

  #ravgsum111 = 0.0
  #for ra111 in range(0,rzrw111):
   #for rb111 in range(0,rzcl111):
    #rimpix111 = rzimcr111[ra111,rb111]
    #ravgsum111 = ravgsum111 + rimpix111

  #rmppp = ravgsum111/(rzrw111*rzcl111)

  #rxavgsum111 = 0.0
  #for rxa111 in range(0,rzrw111):
   #for rxb111 in range(0,rzcl111):
    #rximpix111 = rxzimcr111[rxa111,rxb111]
    #rxavgsum111 = rxavgsum111 + rximpix111

  #rxmppp = rxavgsum111/(rzrw111*rzcl111)

  #ryavgsum111 = 0.0
  #for rya111 in range(0,rzrw111):
   #for ryb111 in range(0,rzcl111):
    #ryimpix111 = ryzimcr111[rya111,ryb111]
    #ryavgsum111 = ryavgsum111 + ryimpix111

  #rymppp = ryavgsum111/(rzrw111*rzcl111)

  #rzavgsum111 = 0.0
  #for rza111 in range(0,rzrw111):
   #for rzb111 in range(0,rzcl111):
    #rzimpix111 = rzzimcr111[rza111,rzb111]
    #rzavgsum111 = rzavgsum111 + rzimpix111

  #rzmppp = rzavgsum111/(rzrw111*rzcl111)

  #r1avgsum111 = 0.0
  #for r1a111 in range(0,rws111,2):
   #for r1b111 in range(0,cls111,2):
    #r1impix111 = imcr111[r1a111,r1b111]
    #r1zim111.append(r1impix111)
    #r1avgsum111 = r1avgsum111 + r1impix111

  #r1mppp = r1avgsum111/(rzrw111*rzcl111)
  #ar1zim111 = np.asarray(r1zim111)  
  #del r1zim111[:]
  #dar1zim111 = np.reshape(ar1zim111, (rzrw111, rzcl111))  #zim111

  #r2avgsum111 = 0.0
  #for r2a111 in range(1,rws111-1):
   #for r2b111 in range(1,cls111-1):
    #r2impix111 = imcr111[r2a111,r2b111] + imcr111[r2a111-1,r2b111-1] + imcr111[r2a111+1,r2b111+1] + imcr111[r2a111-1,r2b111+1] 
    #r2impix111 = r2impix111 + imcr111[r2a111+1,r2b111-1] + imcr111[r2a111-1,r2b111] + imcr111[r2a111,r2b111-1] + imcr111[r2a111+1,r2b111]
    #r2impix111 = r2impix111 + imcr111[r2a111,r2b111+1]
    ##r2zim111.append(r2impix111)
    #r2avgsum111 = r2avgsum111 + (r2impix111/(rzrw111*rzcl111))
    #r2zim111.append(r2impix111/(rzrw111*rzcl111))    

  #r2mppp = r2avgsum111/(rzrw111*rzcl111)
  #ar2zim111 = np.asarray(r2zim111)  
  #del r2zim111[:]
  #dar2zim111 = np.reshape(ar2zim111, (rzrw111, rzcl111)) 

  #r3avgsum111 = 0.0
  #for r3a111 in range(1,rws111-1):
   #for r3b111 in range(1,cls111-1):
    #r3impix111 = (4*imcr111[r3a111,r3b111]) + imcr111[r3a111-1,r3b111-1] + imcr111[r3a111+1,r3b111+1] + imcr111[r3a111-1,r3b111+1] 
    #r3impix111 = r3impix111 + imcr111[r3a111+1,r3b111-1] + (2*imcr111[r3a111-1,r3b111]) + (2*imcr111[r3a111,r3b111-1]) 
    #r3impix111 = r3impix111 + (2*imcr111[r3a111+1,r3b111]) + (2*imcr111[r3a111,r3b111+1])
    ##r3zim111.append(r3impix111)
    #r3avgsum111 = r3avgsum111 + (r3impix111/((rzrw111*rzcl111)+7))
    #r3zim111.append(r3impix111/((rzrw111*rzcl111)+7))    

  #r3mppp = r3avgsum111/(rzrw111*rzcl111)
  #ar3zim111 = np.asarray(r3zim111)  
  #del r3zim111[:]
  #dar3zim111 = np.reshape(ar3zim111, (rzrw111, rzcl111))  

  for x2 in range(et,rows2-et): #and rrow in range(rowws):
   #x2 = x2 + 1
   #rrow = rrow
   #io = io + 1
   #rr.append(x2)
   #if x2 not in rrx:
   #nm = 0
   #d = 0
   for y2 in range(et,cols2-et): #and ccol in range(colls):
     #nm = nm + 1
    #y2 = y2 + 1
    #ccol = ccol
    #ss.append(y2)
    #q = int(img2[x2,y2])
    dist1 = math.sqrt(((x2 - x1)**2)+((y2 - y1)**2))
    #srdist.append(dist1)
    #dist11 = dist1 + 1
    jo = (x2-et,y2-et)
    #area = ((abs(x2 - x1) + 1)*(abs(y2 - y1) + 1))
    #if dist1 <= 11*(math.sqrt(2))
    #for xp2 in range(1,rows2-1):
     #for yp2 in range(1,cols2-1):
      #srdist = math.sqrt(((xp2 - x1)**2)+((yp2 - y1)**2))
      #if srdist <= 5: #*(math.sqrt(2)):
       #srdistl.append(srdist)
       #srdistl.sort()
    
    #srdistlol = srdistl[17]
    #del srdistl[:]
    #for vb in srdistl:
     #if vb not in srl:
      #srl.append(vb) 
        
    #srl.sort()
    #dist2 = (math.sqrt((((x2-1) - (x1-1))**2)+(((y2-1) - (y1-1))**2)))
    #jlo.append(jo)
    #jlo1 = (list(set(jlo).intersection(jjl)))
    #jlen = len(jlo1)
    #del jlo[:]
    #del jlo1[:]
    #for g in range(arr):
     #for h in range(arr[g]):
      #dist2 = (math.sqrt(((g - x1)**2)+((h - y1)**2)))
      #if dist2 <= (7*(math.sqrt(2))):
       #dists2.append(dist2)
    
    
    #for x11 in range(rowws):
     #for y11 in range(colls):
      #for x22 in range(rowws):
       #for y22 in range(colls):
        #dist2 = (math.sqrt((((x22-1) - (x11-1))**2)+(((y22-1) - (y11-1))**2)))
        #if dist2 <= (7*(math.sqrt(2))):
         #dists2.append(arr[x2-1][y2-1])
    #jjls = set(jjl)
    #jjl.sort()
    #dist2 = (math.sqrt((((x2-1) - rrow)**2)+(((y2-1) - ccol)**2)))
    #kl = arr[x2][y2]
    #for r in range(rowws):
     #for s in range(colls):
      #dist2 = (math.sqrt(((r - x1)**2)+((s - y1)**2)))
      #if dist2 <= (7*(math.sqrt(2))):
       #op.append(arr[r][s])

#and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1): #range(arr[: #!= arr[x2-1][y2-1]: #and bisect_right(jjl, jo) != 1: #and not bisect_right(jjl, jo): #jo not in jjls: #not bisect_left(jjl, jo): #jo != arr[x2-1][y2-1]: # jlen == 0
     
     #dist1 <= (5*(math.sqrt(2))) and area <= 11 dist1 <= sz*(math.sqrt(2)) and 

    if dist1 <= (bm*(math.sqrt(2))) and jo != arr[jo[0]][jo[1]]: #and dist11 <= area: dist1 <= sz*(math.sqrt(2)) and area <= sza and 


     #if jo not in op:   
     #q = img2[x2,y2]   #dist2 <= (7*(math.sqrt(2))) and and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1)  
     #q1 = img2[x2-1,y2-1] #(item for sublist in arr for item in sublist) dists2[y2] <= (7*(math.sqrt(2))) and 
     #q2 = img2[x2-1,y2]
     #q3 = img2[x2-1,y2+1] #x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1)
     #q4 = img2[x2,y2-1]
     #q5 = img2[x2,y2+1]
     #q6 = img2[x2+1,y2-1]
     #q7 = img2[x2+1,y2]
     #q8 = img2[x2+1,y2+1]
     #q9 = img2[x2+2,y2+2]
     #q10 = img2[x2-2,y2-2]
     #q11 = img2[x2+1,y2+2]
     #q12 = img2[x2+2,y2+1]
     #q13 = img2[x2,y2+2]
     #q14 = img2[x2+2,y2]
     #q15 = img2[x2-1,y2+2]
     #q16 = img2[x2+2,y2-1]
     #q17 = img2[x2+2,y2-2]
     #q18 = img2[x2-2,y2+2]
     #q19 = img2[x2-2,y2+1]
     #q20 = img2[x2+1,y2-2]
     #q21 = img2[x2,y2-2]
     #q22 = img2[x2-2,y2]
     #q23 = img2[x2-1,y2-2]
     #q24 = img2[x2-2,y2-1]
     #mp = p + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
     #mpp = mp + p9 + p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 + p19 + p20 + p21 + p22 + p23 + p24
  #mp = mp/9
     #mpp = mpp/25
     #mq = q + q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8
     #mqq = mq + q9 + q10 + q11 + q12 + q13 + q14 + q15 + q16 + q17 + q18 + q19 + q20 + q21 + q22 + q23 + q24
     #mq = mq/9
     #mqq = mqq/25
     #sdm2 = float((((q-mq)**2)+((q1-mq)**2)+((q2-mq)**2)+((q3-mq)**2)+((q4-mq)**2)+((q5-mq)**2)+((q6-mq)**2)+((q7-mq)**2)+((q8-mq)**2))/8)
     #imcr2 = imgg2[y2-(et-2):y2+(et-2), x2-(et-2):x2+(et-2)]
     imcr22 = imgg2[y2-(et-1):y2+(et), x2-(et-1):x2+(et)]
     imcr222 = imgg2[y2-et:y2+(et+1), x2-et:x2+(et+1)]
     #imcr22 = cv2.GaussianBlur(imcrr2, (3, 3), 0)

     #imccr2 = imcrr2 - imcr22
     #ret1,th1 = cv2.threshold(imccr2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

     #mmedian1 = cv2.medianBlur(th1,3)
  #mmedian11 = cv2.medianBlur(mmedian1,5)
  #mmedian12 = cv2.medianBlur(mmedian11,5)
  #mmedian13 = cv2.medianBlur(mmedian12,5)
  #mmedian14 = cv2.medianBlur(mmedian13,5)
     #f1 = np.fft.fft2(mmedian1)
     #fshift1 = np.fft.fftshift(f1)
     #imcr2 = 20*np.log(np.abs(fshift1)) #imccr2 
     #rws2, cls2 = imcr2.shape
     rws22, cls22 = imcr22.shape
     rws222, cls222 = imcr222.shape
     #f1 = np.fft.fft2(imcr2)
     #rzimcr222 = cv2.resize(imcr222,(rws222-2, cls222-2), interpolation = cv2.INTER_CUBIC)    #cv2.INTER_CUBIC
     #rxzimcr222 = cv2.resize(imcr222,(rws222-2, cls222-2), interpolation = cv2.INTER_LINEAR) #cv2.INTER_LINEAR
     #ryzimcr222 = cv2.resize(imcr222,(rws222-2, cls222-2), interpolation = cv2.INTER_AREA)   #cv2.INTER_AREA
     #rzzimcr222 = cv2.resize(imcr222,(rws222-2, cls222-2), interpolation = cv2.INTER_NEAREST) #cv2.INTER_NEAREST
     #io = 0
     #rzrw222, rzcl222 = rzimcr222.shape
     #rzrw222 = rws222 - 1 
     #rzcl222 = cls222 - 1
     #avgsum2 = 0.0
     #for aa2 in range(0,rws2):
      #for bb2 in range(0,cls2):
       #impix2 = imcr2[aa2,bb2]
       #avgsum2 = avgsum2 + impix2

     #mq = avgsum2/(rws2*cls2)

     #fshift1 = np.fft.fftshift(f1)

     avgsum22 = 0.0
     for aa22 in range(0,rws22):
      for bb22 in range(0,cls22):
       impix22 = imcr22[aa22,bb22]
       avgsum22 = avgsum22 + impix22

     mqq = avgsum22/(rws22*cls22)

     #iimcr2 = 20*np.log(np.abs(fshift1))

     avgsum222 = 0.0
     for aa222 in range(0,rws222):
      for bb222 in range(0,cls222):
       impix222 = imcr222[aa222,bb222]
       avgsum222 = avgsum222 + impix222

     mqqq = avgsum222/(rws222*cls222)

     #ravgsum222 = 0.0
     #for ra222 in range(0,rzrw222):
      #for rb222 in range(0,rzcl222):
       #rimpix222 = rzimcr222[ra222,rb222]
       #ravgsum222 = ravgsum222 + rimpix222

     #rmqqq = ravgsum222/(rzrw222*rzcl222)

     #rxavgsum222 = 0.0
     #for rxa222 in range(0,rzrw222):
      #for rxb222 in range(0,rzcl222):
       #rximpix222 = rxzimcr222[rxa222,rxb222]
       #rxavgsum222 = rxavgsum222 + rximpix222

     #rxmqqq = rxavgsum222/(rzrw222*rzcl222)

     #ryavgsum222 = 0.0
     #for rya222 in range(0,rzrw222):
      #for ryb222 in range(0,rzcl222):
       #ryimpix222 = ryzimcr222[rya222,ryb222]
       #ryavgsum222 = ryavgsum222 + ryimpix222

     #rymqqq = ryavgsum222/(rzrw222*rzcl222)

     #rzavgsum222 = 0.0
     #for rza222 in range(0,rzrw222):
      #for rzb222 in range(0,rzcl222):
       #rzimpix222 = rzzimcr222[rza222,rzb222]
       #rzavgsum222 = rzavgsum222 + rzimpix222

     #rzmqqq = rzavgsum222/(rzrw222*rzcl222)

     #r1avgsum222 = 0.0
     #for r1a222 in range(0,rws222,2):
      #for r1b222 in range(0,cls222,2):
       #r1impix222 = imcr222[r1a222,r1b222]
       #r1zim222.append(r1impix222)
       #r1avgsum222 = r1avgsum222 + r1impix222

     #r1mqqq = r1avgsum222/(rzrw222*rzcl222)
     #ar1zim222 = np.asarray(r1zim222)  
     #del r1zim222[:]
     #dar1zim222 = np.reshape(ar1zim222, (rzrw222, rzcl222))
  
     #r2avgsum222 = 0.0
     #for r2a222 in range(1,rws222-1):
      #for r2b222 in range(1,cls222-1):
       #r2impix222 = imcr222[r2a222,r2b222] + imcr222[r2a222-1,r2b222-1] + imcr222[r2a222+1,rb222+1] + imcr222[r2a222-1,r2b222+1] 
       #r2impix222 = r2impix222 + imcr222[r2a222+1,r2b222-1] + imcr222[r2a222-1,r2b222] + imcr222[r2a222,r2b222-1] + imcr222[r2a222+1,r2b222]
       #r2impix222 = r2impix222 + imcr222[r2a222,r2b222+1]
       ##r2zim222.append(r2impix222)
       #r2avgsum222 = r2avgsum222 + (r2impix222/(rzrw222*rzcl222))
       #r2zim222.append(r2impix222/(rzrw222*rzcl222))    

     #r2mqqq = r2avgsum222/(rzrw222*rzcl222)
     #ar2zim222 = np.asarray(r2zim222)  
     #del r2zim222[:]
     #dar2zim222 = np.reshape(ar2zim222, (rzrw222, rzcl222)) 

     #r3avgsum222 = 0.0
     #for r3a222 in range(1,rws222-1):
      #for r3b222 in range(1,cls222-1):
       #r3impix222 = (4*imcr222[r3a222,r3b222]) + imcr222[r3a222-1,r3b222-1] + imcr222[r3a222+1,r3b222+1] + imcr222[r3a222-1,r3b222+1] 
       #r3impix222 = r3impix222 + imcr222[r3a222+1,r3b222-1] + (2*imcr222[r3a222-1,r3b222]) + (2*imcr222[r3a222,r3b222-1]) 
       #r3impix222 = r3impix222 + (2*imcr222[r3a222+1,r3b222]) + (2*imcr222[r3a222,r3b222+1])
       ##r3zim222.append(r3impix111)
       #r3avgsum222 = r3avgsum222 + (r3impix222/((rzrw222*rzcl222)+7))
       #r3zim222.append(r3impix222/((rzrw222*rzcl222)+7))    

     #r3mqqq = r3avgsum222/(rzrw222*rzcl222)
     #ar3zim222 = np.asarray(r3zim222)  
     #del r3zim222[:]
     #dar3zim222 = np.reshape(ar3zim222, (rzrw222, rzcl222))  
    #if

      #dist1 = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
      #if dist <= 5*(math.sqrt(2)): 
      #if abs(p-q)<=0 and x2 >= 1 and x2 <= (rows2 - 1) and y2 >= 1 and y2 <= (cols2 - 1) and jo not in jjl and abs(mp-mq)<=0:
      #if x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1): #and dist1 <= (4*(math.sqrt(2))): #and jo not in jjl: #dist1 <= (4*(math.sqrt(2))) and p==q: #abs(p-q)<=0: #and abs(mp-mq)<=0
      #ssim = (((2*mp*mq) + 6.5025)/((mp**2) + (mq**2) + 6.5025)) 
     #msum = 0.0
     #msumx = 0.0
     #msumy = 0.0
     #m1sum2 = 0.0
     #m2sum2 = 0.0
     #msptx = 0.0
     #mspty = 0.0
     #msad2 = 0.0
     #zncc = 0.0
     #ncc2 = 0.0
      #var1 = 0.0
     #for rw1 in range(rws1):
      #for cl1 in range(cls1):
       #imcrp1 = int(imcr1[rw1,cl1])
     #for rw2 in range(0,rws2):
      #for cl2 in range(0,cls2):
       #imcrp1 = int(imcr1[rw1,cl1])
       #imcrp1 = imcr1[rw2,cl2]
       #imcrpl1.append(imcrp1)
       #imcrp2 = imcr2[rw2,cl2]
       #imcrpl2.append(imcrp2)
       #m1sum2 = m1sum2 + ((imcrp1 - mp)**2)
       #m2sum2 = m2sum2 + ((imcrp2 - mq)**2)
       #msum = msum + ((imcrp11 - imcrp22 - mpp + mqq)**2)
       #msum = msum/9
       #msad2 = msad2 + ((abs(imcrp1 - imcrp2 - mp + mq))**2)
       #msad = msad/9
       #msad += ((imcrp1 - imcrp2 - mp + mq))
       #ncc2 = ncc2 + ((imcrp1 - mp)*(imcrp2 - mq))  #(abs(imcrp1 - mp)*abs(imcrp2 - mq))

     #for rwx1 in range(rws1):
      #for clx1 in range(cls1):
       #imcrpx1 = imcr1[rwx1,clx1]
       #imcrpl1.append(imcrpx1)
       #msumx += (imcrpx1**2)
       #msumx = msumx + (imcrpx1 - mp)**2
       #msptx += float(imcrpx1 - mp)

     #for rwy2 in range(rws2):
      #for cly2 in range(cls2):
       #imcrpy2 = imcr2[rwy2,cly2]
       #imcrpl2.append(imcrpy2)
       #msumy += (imcrpy2**2)
       #msumy = msumy + (imcrpy2 - mq)**2
       #mspty += float(imcrpy2 - mq) 

     #msum = 0.0
     #msumx = 0.0
     #msumy = 0.0
     m1sum22 = 0.0
     m2sum22 = 0.0
     rm1sum22 = 0.0
     rm2sum22 = 0.0
     rmx1sum22 = 0.0
     rmx2sum22 = 0.0
     rmy1sum22 = 0.0
     rmy2sum22 = 0.0
     rmz1sum22 = 0.0
     rmz2sum22 = 0.0
     armx1sum22 = 0.0
     armx2sum22 = 0.0
     army1sum22 = 0.0
     army2sum22 = 0.0
     armz1sum22 = 0.0
     armz2sum22 = 0.0
     #m1ssd22 = 0.0
     #m2ssd22 = 0.0
     #msptx = 0.0
     #mspty = 0.0
     #mssd22 = 0.0
     #ncc22 = 0.0
     rzncc22 = 0.0
     rxzncc22 = 0.0
     ryzncc22 = 0.0
     rzzncc22 = 0.0
     ar1zncc22 = 0.0
     ar2zncc22 = 0.0
     ar3zncc22 = 0.0
     zncc22 = 0.0
      #var1 = 0.0
     #for rw1 in range(rws1):
      #for cl1 in range(cls1):
       #imcrp1 = int(imcr1[rw1,cl1])
     for rw22 in range(0,rws22):
      for cl22 in range(0,cls22):
       #imcrp1 = int(imcr1[rw1,cl1])
       imcrp11 = imcr11[rw22,cl22]
       #imcrpl1.append(imcrp1)
       imcrp22 = imcr22[rw22,cl22]
       #imcrpl2.append(imcrp2)

       #rimcrp111 = rzimcr111[rw22,cl22]
       #rimcrp222 = rzimcr222[rw22,cl22]
       #rximcrp111 = rxzimcr111[rw22,cl22]
       #rximcrp222 = rxzimcr222[rw22,cl22]
       #ryimcrp111 = ryzimcr111[rw22,cl22]
       #ryimcrp222 = ryzimcr222[rw22,cl22]
       #rzimcrp111 = rzzimcr111[rw22,cl22]
       #rzimcrp222 = rzzimcr222[rw22,cl22]
       #dar1imcrp111 = dar1zim111[rw22,cl22]
       #dar2imcrp111 = dar2zim111[rw22,cl22]
       #dar3imcrp111 = dar3zim111[rw22,cl22]
       #dar1imcrp222 = dar1zim222[rw22,cl22]
       #dar2imcrp222 = dar2zim222[rw22,cl22]
       #dar3imcrp222 = dar3zim222[rw22,cl22]
 

       m1sum22 = m1sum22 + ((imcrp11 - mpp)**2)
       m2sum22 = m2sum22 + ((imcrp22 - mqq)**2)
       #m1sum22 = m1sum22 + ((imcrp11 - mpp)**2)

       #rm1sum22 = rm1sum22 + ((rimcrp111 - rmppp)**2)
       #rm2sum22 = rm2sum22 + ((rimcrp222 - rmqqq)**2)
       #rmx1sum22 = rmx1sum22 + ((rximcrp111 - rxmppp)**2)
       #rmx2sum22 = rmx2sum22 + ((rximcrp222 - rxmqqq)**2)
       #rmy1sum22 = rmy1sum22 + ((ryimcrp111 - rymppp)**2)
       #rmy2sum22 = rmy2sum22 + ((ryimcrp222 - rymqqq)**2)
       #rmz1sum22 = rmz1sum22 + ((rzimcrp111 - rzmppp)**2)
       #rmz2sum22 = rmz2sum22 + ((rzimcrp222 - rzmqqq)**2)

       #armx1sum22 = armx1sum22 + ((dar1imcrp111 - r1mppp)**2)
       #armx2sum22 = armx2sum22 + ((dar1imcrp222 - r1mqqq)**2)
       #army1sum22 = army1sum22 + ((dar2imcrp111 - r2mppp)**2)
       #army2sum22 = army2sum22 + ((dar2imcrp222 - r2mqqq)**2)
       #armz1sum22 = armz1sum22 + ((dar3imcrp111 - r3mppp)**2)
       #armz2sum22 = armz2sum22 + ((dar3imcrp222 - r3mqqq)**2)

       #msum = msum + ((imcrp11 - imcrp22 - mpp + mqq)**2)
       #m1ssd22 = m1ssd22 + ((imcrp11)**2)
       #m2ssd22 = m2ssd22 + ((imcrp22)**2)
       #m1ssd22 = m1ssd22 + ((imcrp11 - mpp)**2)
       #m2ssd22 = m2ssd22 + ((imcrp22 - mqq)**2)
       #msum = msum/9
       #mssd22 = mssd22 + ((abs(imcrp11 - imcrp22))**2)
       #mssd22 = mssd22 + ((abs(imcrp11 - imcrp22 - mpp + mqq))**2)
       #msad = msad/9
       #msad += ((imcrp1 - imcrp2 - mp + mq))

       #rzncc22 = rzncc22 + ((rimcrp111 - rmppp)*(rimcrp222 - rmqqq))
       #rxzncc22 = rxzncc22 + ((rximcrp111 - rxmppp)*(rximcrp222 - rxmqqq))
       #ryzncc22 = ryzncc22 + ((ryimcrp111 - rymppp)*(ryimcrp222 - rymqqq))
       #rzzncc22 = rzzncc22 + ((rzimcrp111 - rzmppp)*(rzimcrp222 - rzmqqq))
       #ar1zncc22 = ar1zncc22 + ((dar1imcrp111 - r1mppp)*(dar1imcrp222 - r1mqqq))
       #ar2zncc22 = ar2zncc22 + ((dar2imcrp111 - r2mppp)*(dar2imcrp222 - r2mqqq))
       #ar3zncc22 = ar3zncc22 + ((dar3imcrp111 - r3mppp)*(dar3imcrp222 - r3mqqq))

       zncc22 = zncc22 + ((imcrp11 - mpp)*(imcrp22 - mqq))  #(abs(imcrp11 - mpp)*abs(imcrp22 - mqq))

     #msum = 0.0
     #msumx = 0.0
     #msumy = 0.0
     m1sum222 = 0.0
     m2sum222 = 0.0
     #m1ssd222 = 0.0
     #m2ssd222 = 0.0
     #msptx = 0.0
     #mspty = 0.0
     #mssd222 = 0.0
     #zncc = 0.0
     zncc222 = 0.0
      #var1 = 0.0
     #for rw1 in range(rws1):
      #for cl1 in range(cls1):
       #imcrp1 = int(imcr1[rw1,cl1])
     for rw222 in range(0,rws222):
      for cl222 in range(0,cls222):
       #imcrp1 = int(imcr1[rw1,cl1])
       imcrp111 = imcr111[rw222,cl222]
       #imcrpl1.append(imcrp1)
       imcrp222 = imcr222[rw222,cl222]
       #imcrpl2.append(imcrp2)
       #rimcrp222 = rimcr222[rw222,cl222]
       m1sum222 = m1sum222 + ((imcrp111 - mppp)**2)
       m2sum222 = m2sum222 + ((imcrp222 - mqqq)**2)
       #msum = msum + ((imcrp11 - imcrp22 - mpp + mqq)**2)
       #m1ssd222 = m1ssd222 + ((imcrp111)**2)
       #m2ssd222 = m2ssd222 + ((imcrp222)**2)
       #m1ssd222 = m1ssd222 + ((imcrp111 - mppp)**2)
       #m2ssd222 = m2ssd222 + ((imcrp222 - mqqq)**2)
       #msum = msum/9
       #mssd222 = mssd222 + ((abs(imcrp111 - imcrp222))**2)
       #mssd222 = mssd222 + ((abs(imcrp111 - imcrp222 - mppp + mqqq))**2)
       #msad = msad/9
       #msad += ((imcrp1 - imcrp2 - mp + mq))
       zncc222 = zncc222 + ((imcrp111 - mppp)*(imcrp222 - mqqq))  #(abs(imcrp111 - mppp)*abs(imcrp222 - mqqq))
         
           #ssim = ((2*mp*mq
         #msum = msum/9
      #var1 = 0
      #for rw1 in range(rws1):
       #for cl1 in range(cls1):
        #var1 = int(imcr1[rw1,cl1])
        #var1l.append(var1)

      #for rw2 in range(rws2):
       #for cl2 in range(cls2):
        #var2 = int(imcr2[rw2,cl2])
        #var2l.append(var2)
      #msum = msum
     #ssim = ((((2*mp*mq) + 6.5025)*((2*msum) + 58.5225))/(((mp**2) + (mq**2) + 6.5025)*(sdm1 + sdm2 + 58.5225))) 
     #msad = msad/max(
     #msum = msum/((math.sqrt((msumx)*(msumy))))
     #msum = msum/((min(imcrpl1))*(min(imcrpl2)))

     #msad2 = msad2/(rws2*cls2) #*(abs(mp-mq)))
     #msad2 = msad2/(math.sqrt(m1sum2*m2sum2))
     #ncc2 = ncc2/(rws2*cls2)
     #ncc2 = ncc2/(math.sqrt(m1sum2*m2sum2))
     #if abs(mp-mq) != 0:
      #msad2 = msad2/abs(mp-mq)

     #mssd22 = mssd22/(rws22*cls22) #*(abs(mpp-mqq)))
     #mssd22 = mssd22/(math.sqrt(m1ssd22*m2ssd22))
     #ncc22 = ncc22/(rws22*cls22)

     #rzncc22 = rzncc22/(math.sqrt(rm1sum22*rm2sum22))
     #rxzncc22 = rxzncc22/(math.sqrt(rmx1sum22*rmx2sum22))
     #ryzncc22 = ryzncc22/(math.sqrt(rmy1sum22*rmy2sum22))
     #rzzncc22 = rzzncc22/(math.sqrt(rmz1sum22*rmz2sum22))
     #ar1zncc22 = ar1zncc22/(math.sqrt(armx1sum22*armx2sum22))
     #ar2zncc22 = ar2zncc22/(math.sqrt(army1sum22*army2sum22))
     #ar3zncc22 = ar3zncc22/(math.sqrt(armz1sum22*armz2sum22))

     zncc22 = zncc22/(math.sqrt(m1sum22*m2sum22))

     #if abs(mpp-mqq) != 0:
      #msad22 = msad22/abs(mpp-mqq)

     #mssd222 = mssd222/(rws222*cls222) #*(abs(mppp-mqqq)))
     #mssd222 = mssd222/(math.sqrt(m1ssd222*m2ssd222))
     #ncc222 = ncc222/(rws222*cls222)

     zncc222 = zncc222/(math.sqrt(m1sum222*m2sum222))

     #if abs(mppp-mqqq) != 0:
      #msad222 = msad222/abs(mppp-mqqq)

     #ncc = ncc/9*math.sqrt(msumx*msumy)
     #zncc = float((msptx*mspty)/(9*(math.sqrt(sdm1))*(math.sqrt(sdm2))))
     #msum = msum/(rws22*cls22)
     #cssd = m1sum + m2sum - 2*ncc
     #cssd = cssd/25
     #msuml.append(msad2)
     #msumll1.append(mssd22)
     #msumll2.append(mssd222)

     #msuml.append(ncc2)

     #rmsuml.append(rzncc22)
     #rxmsuml.append(rxzncc22)
     #rymsuml.append(ryzncc22)
     #rzmsuml.append(rzzncc22)
     #ar1msuml.append(ar1zncc22)
     #ar2msuml.append(ar2zncc22)
     #ar3msuml.append(ar3zncc22)

     msuml1.append(zncc22)
     msuml2.append(zncc222)

     #del imcrpl1[:]
     #del imcrpl2[:]
     #msumls.append(msad) 
      #print(msum)
      #msum = 0
     #and p1=q1 and p2=q2 and p3=q3 and p4=q4 and p5=q5 and p6=q6 and p7=q7 and p8=q8:
       #def getAverage(imcr, u, v, n):
         #"""img as a square matrix of numbers"""
         #s = 0
         #for i in range(-n, n):
           #for j in range(-n, n):
               #s += imcr[u+i][v+j]
         #return float(s)/(2*n+1)**2

       #def getStandardDeviation(imcr, u, v, n):
        #s = 0
        #avcr = getAverage(imcr, u, v, n)
        #for i in range(-n, n):
           #for j in range(-n, n):
               #s += (imcr[u+i][v+j] - avcr)**2
        #return (s**0.5)/(2*n+1)

       #def zncc(imcr1, imcr2, u1, v1, u2, v2, n):
        #stdDeviation1 = getStandardDeviation(imcr1, u1, v1, n)
        #stdDeviation2 = getStandardDeviation(imcr2, u2, v2, n)
        #avcr1 = getAverage(imcr1, u1, v1, n)
        #avcr2 = getAverage(imcr2, u2, v2, n)

        #s = 0
        #for i in range(-n, n):
           #for j in range(-n, n):
               #s += (imcr1[u1+i][v1+j] - avcr1)*(imcr2[u2+i][v2+j] - avcr2)
        #return float(s)/((2*n+1)**2 * stdDeviation1 * stdDeviation2)

       #if __name__ == "__main__":
   
        #print(zncc(imcr1, imcr2, 1, 1, 1, 1, 1))
        #print(nm)
        #znccl.append((zncc(imcr1, imcr2, 1, 1, 1, 1, 1)))
    #znccpt = int(znccl.index(max(znccl)))    
    #if abs(p-q)<=0 and p1 == q1 and p2 == q2 and p3 == q3 and p4 == q4 and p5 == q5 and p6 == q6 and p7 == q7 and p8 == q8: #and abs(mp-mq)<=0
     dist = math.sqrt(((x2 - x1)**2)+((y2 - y1)**2))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0:
     #dists2.append(dist)
     #if (x1 - x2) <= 0: #and (x1 - x2) <= 0:
      #if dist == 0:
       #dists.append(dist)
      #elif dist != 0:
       #ndist = dist*(-1)
       #dists.append(ndist)
     #elif (y1 - y2) >= 0 and (x1 - x2) >= 0:
     #elif (x1 - x2) > 0:
     dists.append(dist)
     #else:
      #ndist = dist*(-1) 
      #dists.append(ndist)

     jpt = (x2-et,y2-et)
     jptl.append(jpt)
      #xx1.append(x1)
      #yy1.append(y1)

    #elif dist1 <= (6*(math.sqrt(2))) and jo != arr[jo[0]][jo[1]] and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1):

     #imcr2 = imgg2[y2-1:y2+1, x2-1:x2+1]
     #rws2, cls2 = imcr2.shape
 
     #msum = 0
     #msumx = 0
     #msumy = 0
     #msptx = 0
     #mspty = 0
     #msad = 0
     #zncc = 0
     #ncc = 0
      #var1 = 0
     #for rw1 in range(rws1):
      #for cl1 in range(cls1):
       #imcrp1 = int(imcr1[rw1,cl1])
       #for rw2 in range(rws2):
        #for cl2 in range(cls2):
         #imcrp1 = int(imcr1[rw1,cl1])
         #imcrp1 = int(imcr1[rw2,cl2])
         #imcrpl1.append(imcrp1)
         #imcrp2 = int(imcr2[rw2,cl2])
         #imcrpl2.append(imcrp2)
         #msum += int((imcrp1 - imcrp2)**2)
         #msum = msum/9
         #msad += int(abs(imcrp1 - imcrp2))
         #ncc += imcrp1*imcrp2

     #for rwx1 in range(rws1):
      #for clx1 in range(cls1):
       #imcrpx1 = int(imcr1[rwx1,clx1])
       #imcrpl1.append(imcrpx1)
       #msumx += (imcrpx1**2)
       #msptx += float(imcrpx1 - mp)

     #for rwy2 in range(rws2):
      #for cly2 in range(cls2):
       #imcrpy2 = int(imcr2[rwy2,cly2])
       #imcrpl2.append(imcrpy2)
       #msumy += (imcrpy2**2) 
       #mspty += float(imcrpy2 - mq)
 
     #msum = msum/((math.sqrt((msumx)*(msumy))))
     #msum = msum/((min(imcrpl1))*(min(imcrpl2)))
     #msad = msad/9
     #ncc = (ncc/((math.sqrt((msumx)*(msumy)))))
     #zncc = float((msptx*mspty)/(9*(math.sqrt(sdm1))*(math.sqrt(sdm2))))
     #msum = msum/9
     #msuml.append(ncc)
     #del imcrpl1[:]
     #del imcrpl2[:]

     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0:
     #if (x1 - x2) <= 0 and (y1 - y2) >= 0:
      #ndist = dist*(-1)
     #dists.append(dist)
     #elif (x1 - x2) > 0 and (y1 - y2) < 0:
     #else:
      #ndist = dist*(-1) 
      #dists.append(ndist)

     #jpt = (x2-1, y2-1)
     #jptl.append(jpt)
      #xx1.append(x1)
      #yy1.append(y1)

    #elif dist1 <= (7*(math.sqrt(2))) and jo != arr[jo[0]][jo[1]] and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1):

     #imcr2 = imgg2[y2-1:y2+1, x2-1:x2+1]
     #rws2, cls2 = imcr2.shape
 
     #msum = 0
     #msumx = 0
     #msumy = 0
     #msptx = 0
     #mspty = 0
     #msad = 0
     #zncc = 0
     #ncc = 0
      #var1 = 0
     #for rw1 in range(rws1):
      #for cl1 in range(cls1):
       #imcrp1 = int(imcr1[rw1,cl1])
       #for rw2 in range(rws2):
        #for cl2 in range(cls2):
         #imcrp1 = int(imcr1[rw1,cl1])
         #imcrp1 = int(imcr1[rw2,cl2])
         #imcrpl1.append(imcrp1)
         #imcrp2 = int(imcr2[rw2,cl2])
         #imcrpl2.append(imcrp2)
         #msum += int((imcrp1 - imcrp2)**2)
         #msum = msum/9
         #msad += int(abs(imcrp1 - imcrp2))
         #ncc += imcrp1*imcrp2

     #for rwx1 in range(rws1):
      #for clx1 in range(cls1):
       #imcrpx1 = int(imcr1[rwx1,clx1])
       #imcrpl1.append(imcrpx1)
       #msumx += (imcrpx1**2)
       #msptx += float(imcrpx1 - mp)

     #for rwy2 in range(rws2):
      #for cly2 in range(cls2):
       #imcrpy2 = int(imcr2[rwy2,cly2])
       #imcrpl2.append(imcrpy2)
       #msumy += (imcrpy2**2) 
       #mspty += float(imcrpy2 - mq)
 
     #msum = msum/((math.sqrt((msumx)*(msumy))))
     #msum = msum/((min(imcrpl1))*(min(imcrpl2)))
     #msad = msad/9
     #ncc = (ncc/((math.sqrt((msumx)*(msumy)))))
     #zncc = float((msptx*mspty)/(9*(math.sqrt(sdm1))*(math.sqrt(sdm2))))
     #msum = msum/9
     #msuml.append(ncc)
     #del imcrpl1[:]
     #del imcrpl2[:]

     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0:
     #if (x1 - x2) <= 0 and (y1 - y2) >= 0:
      #ndist = dist*(-1)
     #dists.append(dist)
     #elif (x1 - x2) > 0 and (y1 - y2) < 0:
     #else:
      #ndist = dist*(-1) 
      #dists.append(ndist)

     #jpt = (x2-1, y2-1)
     #jptl.append(jpt)
      #xx1.append(x1)
      #yy1.append(y1)

    #elif dist1 <= (8*(math.sqrt(2))) and jo != arr[jo[0]][jo[1]] and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1):

     #imcr2 = imgg2[y2-1:y2+1, x2-1:x2+1]
     #rws2, cls2 = imcr2.shape
 
     #msum = 0
     #msumx = 0
     #msumy = 0
     #msptx = 0
     #mspty = 0
     #msad = 0
     #zncc = 0
     #ncc = 0
      #var1 = 0
     #for rw1 in range(rws1):
      #for cl1 in range(cls1):
       #imcrp1 = int(imcr1[rw1,cl1])
       #for rw2 in range(rws2):
        #for cl2 in range(cls2):
         #imcrp1 = int(imcr1[rw1,cl1])
         #imcrp1 = int(imcr1[rw2,cl2])
         #imcrpl1.append(imcrp1)
         #imcrp2 = int(imcr2[rw2,cl2])
         #imcrpl2.append(imcrp2)
         #msum += int((imcrp1 - imcrp2)**2)
         #msum = msum/9
         #msad += int(abs(imcrp1 - imcrp2))
         #ncc += imcrp1*imcrp2

     #for rwx1 in range(rws1):
      #for clx1 in range(cls1):
       #imcrpx1 = int(imcr1[rwx1,clx1])
       #imcrpl1.append(imcrpx1)
       #msumx += (imcrpx1**2)
       #msptx += float(imcrpx1 - mp)

     #for rwy2 in range(rws2):
      #for cly2 in range(cls2):
       #imcrpy2 = int(imcr2[rwy2,cly2])
       #imcrpl2.append(imcrpy2)
       #msumy += (imcrpy2**2) 
       #mspty += float(imcrpy2 - mq)
 
     #msum = msum/((math.sqrt((msumx)*(msumy))))
     #msum = msum/((min(imcrpl1))*(min(imcrpl2)))
     #msad = msad/9
     #ncc = (ncc/((math.sqrt((msumx)*(msumy)))))
     #zncc = float((msptx*mspty)/(9*(math.sqrt(sdm1))*(math.sqrt(sdm2))))
     #msum = msum/9
     #msuml.append(ncc)
     #del imcrpl1[:]
     #del imcrpl2[:]

     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0:
     #if (x1 - x2) <= 0 and (y1 - y2) >= 0:
      #ndist = dist*(-1)
     #dists.append(dist)
     #elif (x1 - x2) > 0 and (y1 - y2) < 0:
     #else:
      #ndist = dist*(-1) 
      #dists.append(ndist)

     #jpt = (x2-1, y2-1)
     #jptl.append(jpt)
      #xx1.append(x1)
      #yy1.append(y1)



    #elif dist1 <= (8*(math.sqrt(2))) and jo == arr[jo[0]][jo[1]]: #and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1): 
     #dist = np.nan
     #if len(dists) > 0:
      #dist = dists[-1]
      #dists.append(dist)
     #else:
     #if len(dists) == 0:
      #dist = zz[-1] #(sum(zz)/len(zz)) #zz[-1] #0.0 #1.0 #np.nan
     #elif len(dists) > 0:
      #dist = dists[-1]
    
     #dists.append(dist)
     #dist3jpt = (x2, y2)
     #msum = np.nan
     #msuml.append(msum)
     #jpt = (np.nan, np.nan)
     #if len(jptl) == 0 and len(jjl) > 0:
     #jpt = jjl[-1] #(x2-1, y2-1)
     #elif len(jptl) > 0 or len(jjl) == 0:
      #jpt = jptl[-1]

     #jptl.append(jpt)

    #elif dist1 <= (7*(math.sqrt(2))) and x2 >= 1 and x2 <= (rows2-1) and y2 >= 1 and y2 <= (cols2-1): #and len(

     #dist = zz[-1] #(sum(zz)/len(zz)) #zz[-1] #0.0 #1.0 #np.nan
     #dists.append(dist)
     #msum = np.nan
     #msuml.append(msum)
     #jpt = (np.nan, np.nan)
     #jptl.append(jpt)
      #yy2.append(y2)
    #if x2 not in rrx:
     #xx2.append(x2)
     #if dist > 0: #and dist <=50: #((math.sqrt(2))): #or dist == 0:
      #if x1 - x2 <= 0 or y1 - y2 <= 0:
       #ndist = dist*(-1)
       #dists.append(ndist)
       #nx2 = x2*(-1)
       #xx2.append(nx2)
       #ny2 = y2*(-1)
       #yy2.append(ny2)
      #elif x1 - x2 > 0 and y1 - y2 > 0:
       #dists.append(dist)
       #xx2.append(x2)
       #yy2.append(y2) 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q)<=0: #and abs(p1-q1)<=12 and abs(p2-q2)<=12 and abs(p3-q3)<=12 and abs(p4-q4)<=12 and abs(p5-q5)<=12 and abs(p6-q6)<=12 and abs(p7-q7)<=12 and abs(p8-q8)<=12:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 1: 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0: #and dist <=50: #((math.sqrt(2))): #or dist == 0:
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 2:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 1: 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 3:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 1: 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 4:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 1: 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 5:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 6:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 7:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 8:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
    #elif abs(p-q) <= 9:
     #dist = (math.sqrt(((x2 - x1)**2)+((y2 - y1)**2)))
     #if dist >= 0 and dist <=50: #((math.sqrt(2))): #or dist == 0: 
      #if x1 - x2 < 0 or y1 - y2 < 0:
       #ndist = dist*(-1)
       #dists.append(ndist) 
      #elif x1 - x2 >= 0 or y1 - y2 >= 0:
       #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
  #print(dists)
    #jjls.clear()
  #x2m = sum(xx2)/len(xx2)
  #x2md = median(xx2)
  #x2mod = max(set(xx2), key=xx2.count)
  #y2m = sum(yy2)/len(yy2)
  #y2md = median(yy2)
  #y2mod = max(set(yy2), key=yy2.count)
  #zg = (math.sqrt(((x2m - x1)**2)+((y2m - y1)**2)))
  #zgd = (math.sqrt(((x2md - x1)**2)+((y2md - y1)**2)))
  #zgmd = (math.sqrt(((x2mod - x1)**2)+((y2mod - y1)**2)))
  #if sum(dists) > 0: #and len(dists) > 0:
  #if sum(xx2) > 0 and sum(yy2) > 0:
   #mzm = min(dists)
  #znccpt = int(znccl.index(max(znccl)))
  #xx2.append(x2)
  #if len(dists) == 0: #and len(msuml) == 0: #and len(jptl) == 0:
   #znccpt == 0
   #znm = 0
   #del msuml[:]
   #del dists[:]
   #del jptl[:]
  #elif len(dists) != 0: #and len(msuml) != 0: #and len(jptl) != 0:
   #znccpt = int(znccl.index(max(znccl)))
  #for nn in range(msuml):
  #msumla = np.array(msuml) 
  #ii = np.where(msuml == (np.nanmin(msuml)))[0]
  #res_list = [i for i, value in enumerate(msuml) if value == np.nanmin(msuml)]
  print(len(dists))
  print(len(msuml1))
  print(len(jptl))
  #print(np.nanmin(msuml))
  print(np.nanmin(msuml1))
  print(np.nanmin(msuml2))
  #print(np.nanmax(msuml))
  print(np.nanmax(msuml1))
  print(np.nanmax(msuml2))
  #print(np.nanmax(r1msuml))
  #print(np.nanmax(r2msuml)) 
  #maxlist.append(np.nanmax(msuml1))
  #maxlist.append(np.nanmax(msuml2))
  #maxlist.append(np.nanmax(rmsuml))
  #maxlist.append(np.nanmax(rxmsuml))
  #maxlist.append(np.nanmax(rymsuml))
  #maxlist.append(np.nanmax(rzmsuml))
  #maxlist.append(np.nanmax(ar1msuml))
  #maxlist.append(np.nanmax(ar2msuml))
  #maxlist.append(np.nanmax(ar3msuml))
  #maxind = maxlist.index(np.nanmax(maxlist))
  #del maxlist[:]
  #if np.nanmax(msuml1) != np.nan and np.nanmax(msuml2) != np.nan:
  if len(msuml1) > 0 and len(dists) > 0 and len(jptl) > 0: #and np.nanmax(msuml1) != np.nan or np.nanmax(msuml2) != np.nan:
   #if min(msuml) < min(msuml1) and min(msuml) < min(msuml2):
   #if np.isnan(np.nanmax(msuml1)) != False or np.isnan(np.nanmax(msuml2)) != False: 
   if np.nanmax(msuml1) > np.nanmax(msuml2): #and np.nanmax(msuml1) > np.nanmax(r1msuml) and np.nanmax(msuml1) > np.nanmax(r2msuml): #and max(msuml1) > max(msuml2): #np.nanmax(msuml) > np.nanmax(msuml2):
   #if maxind == 0: 
    for ii in range(0,len(msuml1)):
     #if msuml[ii] == min(msuml): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     if msuml1[ii] == np.nanmax(msuml1):
      #vnml.append(ii)
      #vnm = abs(dists[ii])
      vnm = dists[ii]
      vnml.append(vnm)
      #vnmls.append(vnm)
      indexx.append(ii)

   #elif min(msuml1) <= min(msuml) and min(msuml1) <= min(msuml2):
   elif np.nanmax(msuml2) >= np.nanmax(msuml1): #and np.nanmax(msuml2) >= np.nanmax(r1msuml) and np.nanmax(msuml2) >= np.nanmax(r2msuml): #and max(msuml2) >= max(msuml1): #np.nanmax(msuml2) >= np.nanmax(msuml1):
   #elif maxind == 1:
    for ii in range(0,len(msuml2)):
     #if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     if msuml2[ii] == np.nanmax(msuml2):
      #vnml.append(ii)
      #vnm = abs(dists[ii])
      vnm = dists[ii]
      vnml.append(vnm)
      #vnmls.append(vnm)
      indexx.append(ii)

   #elif np.nanmax(r1msuml) >= np.nanmax(msuml1) and np.nanmax(r1msuml) >= np.nanmax(msuml2) and np.nanmax(r1msuml) >= np.nanmax(r2msuml): #and max(msuml2) >= max(msuml1): #np.nanmax(msuml2) >= np.nanmax(msuml1):
   #elif maxind == 2:
    #for ii in range(0,len(rmsuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if rmsuml[ii] == np.nanmax(rmsuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii)

   #elif np.nanmax(r2msuml) >= np.nanmax(msuml1) and np.nanmax(r2msuml) >= np.nanmax(msuml2) and np.nanmax(r2msuml) >= np.nanmax(r1msuml): #and max(msuml2) >= max(msuml1): #np.nanmax(msuml2) >= np.nanmax(msuml1):
   #elif maxind == 3:
    #for ii in range(0,len(rxmsuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if rxmsuml[ii] == np.nanmax(rxmsuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii) 

   #elif maxind == 4:
    #for ii in range(0,len(rymsuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if rymsuml[ii] == np.nanmax(rymsuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii) 

   #elif maxind == 5:
    #for ii in range(0,len(rzmsuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if rzmsuml[ii] == np.nanmax(rzmsuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii)          

   #elif maxind == 6:
    #for ii in range(0,len(ar1msuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if ar1msuml[ii] == np.nanmax(ar1msuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii)
    
   #elif maxind == 7:
    #for ii in range(0,len(ar2msuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if ar2msuml[ii] == np.nanmax(ar2msuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii) 

   #elif maxind == 8:
    #for ii in range(0,len(ar3msuml)):
     ##if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if ar3msuml[ii] == np.nanmax(ar3msuml):
      ##vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      ##vnmls.append(vnm)
      #indexx.append(ii) 

  #elif np.nanmin(msumll1) != np.nan and np.nanmin(msumll2) != np.nan:
  #elif len(msumll1) > 0 and len(dists) > 0 and len(jptl) > 0: #and np.nanmax(msuml1) == np.nan or np.nanmax(msuml2) == np.nan:
   #if min(msuml) < min(msuml1) and min(msuml) < min(msuml2):
   #if np.isnan(np.nanmax(msuml1)) == False or np.isnan(np.nanmax(msuml2)) == False:
    #if np.nanmin(msumll1) < np.nanmin(msumll2): #and max(msuml1) > max(msuml2): #np.nanmax(msuml) > np.nanmax(msuml2):
     #for ii in range(0,len(msumll1)):
      #if msuml[ii] == min(msuml): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
      #if msumll1[ii] == np.nanmin(msumll1):
       #vnml.append(ii)
       #vnm = dists[ii]
       #vnml.append(vnm)
       #vnmls.append(vnm)
       #indexx.append(ii)

    #elif min(msuml1) <= min(msuml) and min(msuml1) <= min(msuml2):
    #elif np.nanmin(msumll2) <= np.nanmin(msumll1): #and max(msuml2) >= max(msuml1): #np.nanmax(msuml2) >= np.nanmax(msuml1):
     #for ii in range(0,len(msumll2)):
      #if msuml1[ii] == min(msuml1): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
      #if msumll2[ii] == np.nanmin(msumll2):
       #vnml.append(ii)
       #vnm = dists[ii]
       #vnml.append(vnm)
       #vnmls.append(vnm)
       #indexx.append(ii)
   

   #elif min(msuml2) < min(msuml) and min(msuml2) < min(msuml1):
   #elif max(msuml2) > max(msuml) and max(msuml2) > max(msuml1):
    #for ii in range(0,len(msuml2)):
     #if msuml2[ii] == min(msuml2): #np.nanmin(msuml): #np.nanmin(msuml) #np.nanmax(msuml):
     #if msuml2[ii] == max(msuml2):
      #vnml.append(ii)
      #vnm = dists[ii]
      #vnml.append(vnm)
      #vnmls.append(vnm)
      #indexx.append(ii)

   if len(vnml) > 1:
    minvnm = min(vnml) #np.nanmin(vnml)
    minind = vnml.index(minvnm)
    mindist = indexx[minind]
    #avgd = sum(vnml)/len(vnml)
    #avgdi = int(avgd)
    #if avgd in dists:
     #mindistw = dists.index(avgd)
     #znm = dists[mindistw]
     #jnm = jptl[mindistw]
    #elif avgdi in dists:
     #mindistwi = dists.index(avgdi)
     #znm = dists[mindistwi]
     #jnm = jptl[mindistwi]
    #elif avgd not in dists:
    jnm = jptl[mindist]
     #znm = avgd
    znm = dists[mindist]
    #if minvnm not in dists:
     #mindist = int(dists.index(minvnm*(-1)))
    #else:
    #mindist = int(dists.index(minvnm))
    #mindistw = dists.index(avgd)
    #jnm = jptl[mindist]
    #znm = avgd
    #jnm = jptl[mindistw]
    #znm = dists[mindist]
    #znm = dists[mindistw]

   elif len(vnml) == 1:
    minvnm = vnml[0]
    minind = vnml.index(minvnm)
    mindist = indexx[minind]
    #if minvnm not in dists:
     #mindist = int(dists.index(minvnm*(-1)))
    #else:
    #mindist = int(dists.index(minvnm))
    jnm = jptl[mindist]
    znm = dists[mindist] 

   #elif len(vnml) == 0:
    #if len(dists) 
    #jnm = jjl[-1] #jptl[-1]
    #for h in range(len(dists)):
     #if dists[h] == np.nan:
      #dl = []
      #dl.append(dists[h])
      
    #znm = zz[-1] #dists[-1] #abs(np.nanmax(dists) - np.nanmin(dists)) #(sum(dists)/len(dists)) #[-1] #min(dists) #np.nanmin(dists) #np.nanmax(dists)

  #elif len(msuml) == 0 and len(dists) == 0 and len(jptl) == 0:
   #jnm = jjl[0]
   #dist3 = dist = (math.sqrt(((abs(x2 - x1))**2)+((abs(y2 - y1))**2)))
   #vv = 118 #*(x1-1)
   #znm = zz[-1] #abs(zz[0] + zz[-1]) #median(zz) #(sum(zz)/len(zz))#zz[-1] #abs(np.nanmax(dists) - np.nanmin(dists))
  #if len(vnml) == 1:
   #if len(dists) == 0:
   #ww = rowws
   #wwf = ww*3
   #ff = 1
   #cc = rowws - 1
   #dd = rowws + 1
   #avgznm = zz[-ww] + zz[-cc] + zz[-dd] #zz[-ff] +
   #jnm = jjl[-ww]
   #znm = avgznm/3
   #znm = zz[-ww] #zz[-118] #(sum(zz)/len(zz)) #zz[-1] #0.0 #1.0 #np.nan
   #elif len(dists) > 0:
    #dist = dists[-1]

   #minvnm = vnml[0]
   #minind = int(vnml.index(minvnm))
   #mindist = indexx[minind]
  #elif len(vnml) > 1:
   #vnmls.sort()
   #msumls.sort()
   #vnms = msumls[1]
   #minvnm = vnmls[1]
   #minvnm = np.nanmin(vnml)
   #minind = int(vnml.index(minvnm))
   #mindist = indexx[minind]
   
   #jnm = jptl[mindist]
   #znm = dists[mindist]

  #elif len(msuml) == 0 and len(dists) == 0 and len(jptl) == 0:
   #jnm = jjl[-1] #(np.nan, np.nan)
   #znm = zz[-1] #np.empty #zz[-1]
   #mindist = 2

  #if len(vnml) == 1:
   #minvnm = vnml[0]
   #minind = int(vnml.index(minvnm))
   #mindist = indexx[minind]
  #elif len(vnml) > 1:
   #minvnm = np.nanmin(vnml)
   #minind = int(vnml.index(minvnm))
   #mindist = indexx[minind]
   #else:
  #ii = 0
  #while ii < len(msuml):
   #if msuml[ii] == min(msuml):
    #vnm = dists[ii]
    #vnml.append(vnm)
   #else:
    #ii = ii + 1
  
  #if len(vnml) > 1:
   #minvnm = np.nanmin(vnml)
   #minind = int(vnml.index(minvnm))
   #mindist = indexx[minind]
   #if minvnm not in dists:
    #mindist = int(dists.index(minvnm*(-1)))
   #else:
   #mindist = int(dists.index(minvnm))
  #elif len(vnml) == 1:
   #minvnm = vnml[0]
   #minind = int(vnml.index(minvnm))
   #mindist = indexx[minind]
   #if minvnm not in dists:
    #mindist = int(dists.index(minvnm*(-1)))
   #else:
   #mindist = int(dists.index(minvnm))
  #elif len(vnml) == 0:
   #minvnm = 0
   
  #if len(vnml) == 1:
   #minmsum = int(msuml.index(np.nanmin(msuml)))
   #jnm = jptl[minmsum]
   #znm = dists[minmsum]
  #else:
   #for tt in range(
   #minvnm = min(vnml)
   #minmsum = int(vnml.index(minvnm))
   #jnm = jptl[minmsum]
   #znm = dists[minmsum]
  #ii = [i for i, x in enumerate(msuml) if x == (np.nanmin(msuml))]
  #for vv in range(len(ii)):
   #vnm = dists[vv]
   #vnml.append(vnm)
  
  #ab = nan
  #if ab not in vnml:
   #minvnm = min(vnml)
  #else:
   #minvnm = np.nanmin(vnml)
  #minmsum = int(msuml.index(np.nanmin(msuml)))
  #mindist = int(dists.index(minvnm))
  
  
  #if mindist == minmsum:
  #if mindist >= len(dists) and mindist >= len(msuml) and mindist >= len(jptl): #and 
   #jnm = jjl[-1] #(np.nan, np.nan)
   #znm = 1 #zz[-1] #(sum(zz)/len(zz)) #zz[-1] #np.nan
  #elif mindist < len(dists) and mindist < len(msuml) and mindist < len(jptl): #and 
   #jnm = jptl[mindist]
   #znm = dists[mindist]
  #print(znm)
  n = n + 1
  print(n)
  print(znm)
  #else:
   #jnm = 0
   #znm = 0
  del msuml[:]
  del rmsuml[:]
  del rxmsuml[:]
  del rymsuml[:]
  del rzmsuml[:]
  del ar1msuml[:]
  del ar2msuml[:]
  del ar3msuml[:]
  del r1msuml[:]
  del r2msuml[:]
  del msuml1[:]
  del msuml2[:]
  del msumll1[:]
  del msumll2[:]
  del msumls[:]
  del dists[:]
  del dists2[:]
  del jptl[:]
  del vnml[:]
  del vnmls[:]
  del indexx[:]
  #del dists2[:]
  #jjls.clear()
  #del op[:]
  #del ii[:]
   #print(znm)
   #else:
    #znm = 0
   #jnm = jptl[minmsum]
   #xnm = x2#[io]
   #ynm = yy2[znccpt]
   #xnm = rr[znccpt]
   #ynm = ss[znccpt]
  #print(znm)
   #zm = sum(dists)/len(dists)
   #new.append(zm)
   #mz = np.median(dists)
   #new.append(mz)
   #zmod = max(set(dists), key=dists.count)
   #zmood = min(set(dists), key=dists.count)
   #dmt = abs(zmod - zmood)
   #new.append(zmod)
   #new.append(zmood)
   #newz = min(new)
   #newm = sum(new)/len(new)
   #zdif = (zmod - zm)
   #zmd = (((np.median(dists))+zm+zmod)/3)
   #zav = (zm + zmod)/2
   #zmod = max(set(dists), key=dists.count)
   #x2m = sum(xx2)/len(xx2)
   #x2md = median(xx2)
   #x2mod = max(set(xx2), key=xx2.count)
   #y2m = sum(yy2)/len(yy2)
   #y2md = median(yy2)
   #y2mod = max(set(yy2), key=yy2.count)
   #zg = (math.sqrt(((x2m - x1)**2)+((y2m - y1)**2)))
   #zgd = (math.sqrt(((x2md - x1)**2)+((y2md - y1)**2)))
   #zgmd = (math.sqrt(((x2mod - x1)**2)+((y2mod - y1)**2)))
   #if (max(dists)-min(dists)) > 0:
  zmnorm = znm #)/median(dists))#(np.std(dists,ddof = 0)) #/(max(dists))#-min(dists))
   #rzmnorm = 1/zmnorm
   #zmsd = zm/(np.std(dists,ddof = 0))
   #rzmsd = 1/zmsd
  if zmnorm > 0 or zmnorm < 0:
   zz.append(1/zmnorm)
  elif zmnorm == 0:
   zz.append(1)
  jjl.append(jnm)
  print(jnm)
  #jjls = set(jjl)
  #jjl.sort()
  #for i in range(x1):
   #for j in range(y1):
  #arr[jnm[0]][jnm[1]] = (jnm[0], jnm[1])
  arr[jnm[0]][jnm[1]] = jnm
  #arr[jnm[0]][jnm[1]] = 1
  print(arr[jnm[0]][jnm[1]])
  #arr.sort()
  #jjl.sort() #(key=lambda k: [k[1], k[0]])
  #jjl = sorted(jjl, key=lambda element: (element[0], element[1]))
  #sort1 = sorted(jjl,key=itemgetter(1))
  #rrx.append(xnm)
  #ssx.append(ynm)
  #elif sum(dists) == 0: #and len(dists) == 0:
  #elif sum(xx2) == 0 and sum(yy2) == 0:
   #zmnorm = 0
   #rzmnorm = zmnorm
   #zmsd = zm#/(np.std(dists,ddof = 1))
   #zz.append(rzmnorm)
  #zmd = median(dists)
  #zmod = max(set(dists), key=dists.count)
  #ze = min(dists)
  #z = min(dists,key = abs)
  #zx = abs(min(dists,key = abs))
  #if zx >= 15:
    #zx = 1
  #nz = min(dists)
  #nz1 = nz*(-1)
  #z1 = math.sqrt(zx)
  #zc = np.cbrt(zx)
  #zr = zx**(1/float(11.0))
  #zr1 = zx**(3/float(2.0))
  #z2 = math.exp(z)
  #z3 = math.exp(-(z))
  #z4 = (1/z)
  #z5 = math.exp(-(1/z))
  #z6 = (1/((zx)))
  #z7 = (1/(math.sqrt(z)))
  #z8 = zx**2
  #z9 = z**3
  #z10 = z*2
  #z11 = z*3
  #if ze > 10:
   #ze = 10 
  #zz.append(ze)
  #zz.append(zmnorm)
  #if ze == 2 or ze >= 5:
   #zz.append(np.nan)#**1.75)
   #r1r.append(np.nan)
   #s1s.append(np.nan)
  #elif ze > 2 and ze < 5:
   #zee = math.sqrt(ze)
   #zec = np.cbrt(ze)
   #zeee = ze**1.5
   #zz.append(ze)#**(1.75))
   #r1r.append(x1)
   #s1s.append(y1)
  #zza = np.array(zz).reshape(rr,ss)
  #z12 = (z**(1./9.))
  #if z < nz:
   #zz.append(z)
  #elif z >= nz:
   #zz.append(nz1)
  
  #a = int(dists.index(min(dists,key = abs)))
  #b = int(dists.index(min(dists,key = abs)))
  #c = int(dists.index(min(dists,key = abs)))
  #d = int(dists.index(min(dists,key = abs)))
  #ax = int(dists.index(abs(min(dists,key = abs))))
  #bx = int(dists.index(abs(min(dists,key = abs))))
  #cx = int(dists.index(abs(min(dists,key = abs))))
  #dx = int(dists.index(abs(min(dists,key = abs))))
  #dists.clear()
  #r = xx1[a]
  #s = yy1[b]
  #t = xx2[c]
  #u = yy2[d]
  #rx = xx1[ax]
  #sx = yy1[bx]
  #tx = xx2[cx]
  #ux = yy2[dx]
  #rr.append(x1)
  #ss.append(y1)
  #tt.append(t)
  #uu.append(u)
  #rrx.append(rx)
  #ssx.append(sx)
  #ttx.append(tx)
  #uux.append(ux)
  #del new[:]
  #del dists[:]
  #del xx1[:]
  #del yy1[:]
  #del xx2[:]
  #del yy2[:]
  #del znccl[:]
  #del jptl[:]
  #del msuml[:]
  #del rr[:]
  #del ss[:]
 #ss.append(y1)
 #r1r.append(x1)
 #s1s.append(y1)

#for x3 in range(rows3-2):
 #x3 = x3 + 1
 #rr.append(x1)
 
 #for y3 in range(cols3-2):
 
  #y3 = y3 + 1
  #pp = int(img3[x3,y3])
  #if x3 >= 1 and x3 <= (rows3 - 1) and y3 >= 1 and y3 <= (cols3 - 1):
   #pp = int(img1[x3,y3])
   #pp1 = int(img1[x3-1,y3-1])
   #pp2 = int(img1[x3-1,y3])
   #pp3 = int(img1[x3-1,y3+1])
   #pp4 = int(img1[x3,y3-1])
   #pp5 = int(img1[x3,y3+1])
   #pp6 = int(img1[x3+1,y3-1])
   #pp7 = int(img1[x3+1,y3])
   #pp8 = int(img1[x3+1,y3+1])
   #mpp = pp + pp1 + pp2 + pp3 + pp4 + pp5 + pp6 + pp7 + pp8
  #for x4 in range(rows4-2):
   #x4 = x4 + 1
   #for y4 in range(cols4-2):
    #y4 = y4 + 1
    #qq = int(img4[x4,y4])
    #if x4 >= 1 and x4 <= (rows4 - 1) and y4 >= 1 and y4 <= (cols4 - 1):
     #qq = int(img2[x4,y4])
     #qq1 = int(img2[x4-1,y4-1])
     #qq2 = int(img2[x4-1,y4])
     #qq3 = int(img2[x4-1,y4+1])
     #qq4 = int(img2[x4,y4-1])
     #qq5 = int(img2[x4,y4+1])
     #qq6 = int(img2[x4+1,y4-1])
     #qq7 = int(img2[x4+1,y4])
     #qq8 = int(img2[x4+1,y4+1])
     #mqq = qq + qq1 + qq2 + qq3 + qq4 + qq5 + qq6 + qq7 + qq8        
    #if abs(pp-qq)<=0 and pp1==qq1 and pp2==qq2 and pp3==qq3 and pp4==qq4 and pp5==qq5 and pp6==qq6 and pp7==qq7 and pp8==qq8: #abs(mpp-mqq)<=0
     #disty = (math.sqrt(((x4 - x3)**2)+((y4 - y3)**2)))
     #if dist >= (math.sqrt(2)): 
      #dists.append(dist)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx2.append(x2)
      #yy2.append(y2)
     #if disty >= 0: 
      #distys.append(disty)
      #xx1.append(x1)
      #yy1.append(y1)
      #xx4.append(x4)
      #yy4.append(y4)
     #if dist > 0: #and dist <=50: #((math.sqrt(2))): #or dist == 0:
      #if x3 - x4 <= 0 or y3 - y4 <= 0:
       #ndisty = disty*(-1)
       #distys.append(ndisty)
       #nx4 = x4*(-1)
       #xx4.append(nx4)
       #ny4 = y4*(-1)
       #yy4.append(ny4)
      #elif x3 - x4 >= 0 and y3 - y4 >= 0:
       #distys.append(disty)
       #xx4.append(x4)
       #yy4.append(y4)

  #x4m = sum(xx4)/len(xx4)
  #x4md = median(xx4)
  #x4mod = max(set(xx4), key=xx4.count)
  #y4m = sum(yy4)/len(yy4)
  #y4md = median(yy4)
  #y4mod = max(set(yy4), key=yy4.count)
  #zgg = (math.sqrt(((x4m - x3)**2)+((y4m - y3)**2)))
  #zggd = (math.sqrt(((x4md - x3)**2)+((y4md - y3)**2)))
  #zggmd = (math.sqrt(((x4mod - x3)**2)+((y4mod - y3)**2)))
  #if sum(distys) > 0: #and len(distys) > 0:
  #if sum(xx4) > 0 and sum(yy4) > 0:
   #mzmm = min(distys)
   #zmm = sum(distys)/len(distys)
   #neww.append(zmm)
   #mmz = np.median(distys)
   #neww.append(mmz)
   #zmmod = max(set(distys), key=distys.count)
   #zmmood = min(set(distys), key=distys.count)
   #dmtt = abs(zmmod - zmmood)
   #neww.append(zmmod)
   #neww.append(zmmood)
   #newwz = min(neww)
   #newwm = sum(neww)/len(neww)
   #zdiff = (zmmod - zmm)
   #zmmd = (((np.median(distys))+zmm+zmmod)/3)
   #zavv = (zmm + zmmod)/2
   #zmmod = max(set(distys), key=distys.count)
   #x4m = sum(xx4)/len(xx4)
   #x4md = median(xx4)
   #x4mod = max(set(xx4), key=xx4.count)
   #y4m = sum(yy4)/len(yy4)
   #y4md = median(yy4)
   #y4mod = max(set(yy4), key=yy4.count)
   #zgg = (math.sqrt(((x4m - x3)**2)+((y4m - y3)**2)))
   #zggd = (math.sqrt(((x4md - x3)**2)+((y4md - y3)**2)))
   #zggmd = (math.sqrt(((x4mod - x3)**2)+((y4mod - y3)**2)))
   #zmmnorm = zmm #)/median(distys)) #(np.std(distys,ddof = 0)) #/(max(distys))#-min(distys))
   #rzmmnorm = 1/zmmnorm
   #zmmsd = zmm/(np.std(distys,ddof = 0))
   #rzmmsd = 1/zmmsd
   #znz.append(zmmnorm)
  #elif sum(distys) == 0: #and len(distys) == 0:
  #elif sum(xx4) == 0 and sum(yy4) == 0:
   #zmm = 0
  #if (max(distys)-min(distys)) > 0:
   #zmmnorm = zmm/(max(distys)-min(distys))
   #rzmmnorm = 1/zmmnorm
   #zmmsd = zmm/(np.std(distys,ddof = 1))
  #elif (max(distys)-min(distys)) == 0:
   #zmmnorm = zmm
   #rzmmnorm = zmmnorm
   #znz.append(zmmnorm)
   #zmmsd = zmm#/(np.std(distys,ddof = 1)) 
  #zmmd = median(distys)
  #zmmod = max(set(distys), key=distys.count)
  #znz.append(zmmnorm)
  #del neww[:]
  #del distys[:]
  #del xx1[:]
  #del yy1[:]
  #del xx4[:]
  #del yy4[:]

#print(len(rr))
#print(len(ss))
#print(len(zz))
#print(jjl)
print(arr)
print(max(zz))
for k in range(len(zz)):
  #dz = abs(znz[k] - zz[k])
  #dz = dz*2
  #dz = #(znz[k] + zz[k])/2
  #if dz
  dzz.append(zz[k])
  #if znz[k] <= zz[k]:
   #dzz.append(zz[k])
  #elif znz[k] > zz[k]:
   #dzz.append(znz[k])

#zaz = np.asarray(zz)
#zaza = np.reshape(zaz, (rows3, cols3))
#zbz = np.asarray(znz)
#zbzb = np.reshape(zbz, (rows4, cols4))

#for xr1 in range(rows3):
 #for yr1 in range(cols3):
  #pz = int(zaza[xr1,yr1])
  #for xs1 in range(rows4):
   #for ys1 in range(cols4):
    #qz = int(zbzb[xs1,ys1])
    #if abs(pz-qz) <= 0:
     #edist = (math.sqrt(((xs1 - xr1)**2)+((ys1 - yr1)**2)))
     #if edist >=0:
      #edisty.append(edist)

  
  #if sum(edisty) > 0: #and len(edisty) > 0:
   #ez1 = min(edisty)
   #ez1 = sum(edisty)/len(edisty)
   #ez2 = median(edisty)
   #ez3 = max(set(edisty), key=edisty.count)
   #ezz.append(ez1)
  #elif sum(edisty) == 0: #and len(edisty) == 0:
   #ez1 = 0
   #ezz.append(ez1)
  #del edisty[:]

zzn = np.asarray(dzz)
#def nan_helper(zzn):
  #return np.isnan(zzn), lambda z: z.nonzero()[0]

#nans, x= nan_helper(zzn)
#zzn[nans]= np.interp(x(nans), x(~nans), zzn[~nans])
#for i in range(1,len(zz)):
  #if math.isnan(zz[i]):
    #zz[i] = zz[i-1]

zz2d = np.reshape(zzn, (rowws, colls))
zz2d = zz2d[::-1]
#xl = np.arange(0, zz2d.shape[1])
#yl = np.arange(0, zz2d.shape[0])
#mask invalid values
#zz2d = np.ma.masked_invalid(zz2d)
#xxl, yyl = np.meshgrid(xl, yl)
#get only the valid values
#x1l = xxl[~zz2d.mask]
#y1l = yyl[~zz2d.mask]
#nzz2d = zz2d[~zz2d.mask]
#zzz2d = interpolate.interp2d(xl, yl, zz2d, kind='cubic')
#n1zz2d = interpolate.griddata((x1l, y1l), nzz2d.ravel(),(xxl, yyl),method='nearest')
                          
#nanss = np.zz2d(np.where(np.isnan(X))).T
#notnans = np.zz2d(np.where(~np.isnan(X))).T
#for pl in nanss:
    #X[pl[0],pl[1]] = sum( X[ql[0],ql[1]]*np.exp(-(sum((pl-ql)**2))/2) for ql in notnans )                             

#n1zz2d = cv2.GaussianBlur(n1zz2d, (15, 15), 0)
zz2d = cv2.GaussianBlur(zz2d, (hf, hf), 0)
#szx = (sz*2) + 1
#zz2d = cv2.GaussianBlur(zz2d, (szx, szx), 0)
#zz2d = cv2.blur(zz2d,(91,91))
#xxn = np.array(xx)
#yyn = np.array(yy)
#xxv = np.asarray(rr)
#yyv = np.asarray(ss)
#zzv = np.asarray(zz)

#zza = np.array(zz).reshape(len(rr),len(ss))    
#rrn = rr.sort()
#ssn = ss.sort()

  
  #a1 = rr.index(xv)
  #b2 = ss.index(yv)
  #return
fn = get_sample_data("Dubey Rt 15cr1.png", asfileobj=False)
gmi = read_png(fn) 
fig = plt.figure()#figsize=(6,6))
#ax = fig.add_subplot(111,projection = '3d')
#ax = fig.gca(projection='3d')
#xv, yv = np.meshgrid(colls, rowws[::-1])
xv, yv = np.ogrid[0:gmi.shape[0], 0:gmi.shape[1]]
ax = fig.gca(projection='3d')
#x1v, y1v = np.meshgrid(s1s,r1r[::-1])
#xv, yv = np.meshgrid(s1s,r1r[::-1])
#def to_matrix(l, n):
    #return [zzn[i:i+n] for i in xrange(0, len(zzn), n)]
#def f(xv,yv):
  #return zz[(int(xv+(((len(xv)-1)*xv)+yv))/2)-1]
#zv = np.array(zz)
#ax.elev= 75
#ax.set_zlim3d(1,100)
#ax.set_aspect('equal')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_zscale('log')
#ax.scatter3D(rowws, colls, zz2d, marker = '.')
#ax.plot_wireframe(xv, yv, zz2d)
#zv = to_matrix(xv,yv)
ax.plot_surface(xv,yv,zz2d, facecolors=gmi)#, rstride = 1, cstride = 1)#, rcount=14400, ccount=14400)
#ax.plot_trisurf(xv, yv, zz2d)#,cmap='viridis', edgecolor='none')
#ax.contour3D(xv, yv, zz2d)#, 50, cmap='binary')
#ax.plot_wireframe(xv, yv, zz2d, rstride = 1, cstride = 1)#,rcount=np.inf, ccount=np.inf, color='black')                
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
