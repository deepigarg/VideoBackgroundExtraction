import cv2 
import numpy as np
import statistics
from datetime import datetime as dt

def variance(freq, keys, start, end): #function to find total frequency and variance for otsu
    V = 0.0
    n = 0
    mean = 0.0
    for i in keys[start:end+1]:
        mean+= freq[i]*i
        n+=freq[i]
    mean = mean/n
    for i in keys[start:end+1]:
        V+= freq[i]*(i-mean)*(i-mean)
    V = V/n
    return n,V

def otsu(img):
    rows = len(img)
    cols = len(img[0])
    freq = {} # dictionary with frequencies
    for i in img:
        for j in i:
            j2 = float((int(j*100))/100.0)
            freq[j2] = freq.get(j2, 0) + 1

    keys = list(freq.keys())
    keys.sort()

    i0 = variance(freq, keys, 0,0)
    i1 = variance(freq, keys, 1,len(keys))
    MIN = i0[0]*i0[1] + i1[0]*i1[1]
    thresh = 1

    for t in range(2,len(keys)):
        i0 = variance(freq, keys, 0,t-1)
        i1 = variance(freq, keys, t,len(keys))
        val = i0[0]*i0[1] + i1[0]*i1[1]
        if(val<MIN):
            MIN = val
            thresh = keys[t]

    return thresh

def extract_bg(img, imgcol, thresh): # extract BG given bw diff frame, colored frame and otsu thresh
    rows = len(img)
    cols = len(img[0])
    for i in range(rows):
        for j in range(cols):
            if(img[i][j]<thresh):
                imgcol[i][j] = [0,255,255]


t1 = dt.now()
# Read the video
cam = cv2.VideoCapture("cv3_video1.mp4") 
 
currentframe = 0
meanMat = [] # avg frame
medMat = [] # median frame
modeMat = [] # mode frame
aggMat = [] # aggregate matrix to find median and mode
rows=0
cols=0
while(True): # read the video - store each frame and update mean, median, mode matrices
    ret,frame = cam.read()
    if ret: 

        name = './data/frame' + str(currentframe) + '.jpg'
        if currentframe==0: # initialize matrices in case of the first frame
            rows = len(frame)
            cols = len(frame[0])
            meanMat = np.zeros((rows, cols, 3,))
            medMat = np.zeros((rows, cols, 3,))
            modeMat = np.zeros((rows, cols, 3,))
            for i in range(rows):
                aggMat.append([])
                for j in range(cols):
                    aggMat[i].append([])
                    for k in range(3):
                        aggMat[i][j].append([])

        for i in range(rows):
            for j in range(cols):
                for k in range(3):
                    aggMat[i][j][k].append(frame[i][j][k])
                    meanMat[i][j][k]+=frame[i][j][k]

        cv2.imwrite(name, frame) 
        currentframe += 1
    else: 
        break

print(dt.now()-t1, ": Frames created. Aggregate matrix got.")

numFrames = currentframe

# create final mean, median and mode frames
for i in range(rows):
    for j in range(cols):
        for k in range(3):
            (aggMat[i][j][k]).sort()            
            meanMat[i][j][k] = meanMat[i][j][k]/numFrames
            medMat[i][j][k] = aggMat[i][j][k][int(numFrames/2)]
            modeMat[i][j][k] = max(aggMat[i][j][k], key = (aggMat[i][j][k]).count)

# save mean, median, mode frames
cv2.imwrite('./ans/meanFrame.jpg', meanMat)
cv2.imwrite('./ans/medianFrame.jpg', medMat)
cv2.imwrite('./ans/modeFrame.jpg', modeMat)
print(dt.now()-t1, ": Mean, median, mode frames created")

thresh1 = 20
thresh2 = 20
thresh3 = 20
for i in range(numFrames):
    path = './data/frame' + str(i) + '.jpg'
    frm = cv2. imread(path)
    frm2 = cv2. imread(path)
    frm3 = cv2. imread(path)

    # frame - mean/median/mode frame
    meanfr = np.absolute(frm - meanMat)
    modefr = np.absolute(frm - modeMat)
    medfr = np.absolute(frm - medMat)
    # convert to grayscale
    meanfr32 = np.float32(meanfr)
    modefr32 = np.float32(modefr)
    medfr32 = np.float32(medfr)
    meanfrbw = cv2.cvtColor(meanfr32, cv2.COLOR_BGR2GRAY)
    modefrbw = cv2.cvtColor(modefr32, cv2.COLOR_BGR2GRAY)
    medfrbw = cv2.cvtColor(medfr32, cv2.COLOR_BGR2GRAY)
    # apply otsu
    if(i==0):
        print(dt.now()-t1, ": Otsu start")
        thresh1 = otsu(meanfrbw)
        thresh2 = otsu(modefrbw)
        thresh3 = otsu(medfrbw)
        print(thresh1, thresh2, thresh3)
        print(dt.now()-t1, "Otsu end")
    # extract bg using otsu thresh
    extract_bg(meanfrbw, frm, thresh1)
    extract_bg(modefrbw, frm2, thresh2)
    extract_bg(medfrbw, frm3, thresh3)
    # save extracted frame
    cv2.imwrite('./ans/mean'+str(i)+'.jpg', frm)
    cv2.imwrite('./ans/mode'+str(i)+'.jpg', frm2)
    cv2.imwrite('./ans/median'+str(i)+'.jpg', frm3)

cam.release() 
cv2.destroyAllWindows() 
print(dt.now()-t1, ": Done")
