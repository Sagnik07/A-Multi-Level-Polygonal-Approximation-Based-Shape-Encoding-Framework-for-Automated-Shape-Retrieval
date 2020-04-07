import numpy as np
import math
import sys, os, csv, cv2
import ShapePolygonApproximation

def getClockwiseAngle(x0, y0, x1, y1, x2, y2):
    # clockwise angle of vector (p0-p1) to align with (p1-p2) with respect to center at p1 
    angle = 0.0
    dx1 = x0 - x1
    dx2 = x2 - x1
    dy1 = y0 - y1
    dy2 = y2 - y1


    angle = math.degrees(math.atan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2))
    
    if(angle < 0):
        angle += 360
   
        
    return angle

def getFeatureVector(imageFilePath):
    top, bottom = 100//2, 100//2
    left, right = 100//2, 100//2
    img = cv2.imread(imageFilePath)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    img2 = img.copy();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(thresh_gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #print("Shape of contour array: ", cnt.shape) 
    cnt = np.array(cnt)
    cnt = cnt.reshape(len(cnt), 2).tolist()
    
    data = []
    numberOfKeyPts = 4
     
    while(numberOfKeyPts <= 6):
        keyPts, lineThickness = ShapePolygonApproximation.getKeyPoints(cnt, numberOfKeyPts)
        # print("keyPts = ", keyPts)
        c = np.array(keyPts)
        print(c)
        subvect = []
        for k in range(numberOfKeyPts):
            q = (k+1) % len(keyPts)
            p = (k+2) % len(keyPts)
            a = int(getClockwiseAngle(keyPts[k][0], keyPts[k][1], keyPts[q][0], keyPts[q][1], keyPts[p][0], keyPts[p][1]))
            subvect.append(a)
            print(a)
        
        subvect.sort()
          
        print(subvect) 
        for featurePt in subvect:
            data.append(featurePt)
        numberOfKeyPts += 2 
    print(data)
    return data

def main():
    top, bottom = 100//2, 100//2
    left, right = 100//2, 100//2
    
    data_path = './data'
    print(data_path)
    data_dir_list = os.listdir(data_path)
    print(data_dir_list)
    imageCount = 0
    with open('dataset.csv', 'w') as csvFile:
        csvFile.close()
    
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print(dataset)
        print ('Loaded the images of dataset: '+'{}\n'.format(dataset))
        classMemberCnt = 0
        for img in img_list:
            classMemberCnt += 1
            resImagePath = data_path + '/'+ dataset + '/'+ img
            print(resImagePath)
            img = cv2.imread(resImagePath)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
            img2 = img.copy();
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, h = cv2.findContours(thresh_gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            #print("Shape of contour array: ", cnt.shape) 
            cnt = np.array(cnt)
            cnt = cnt.reshape(len(cnt), 2).tolist()
    
            data = []
            numberOfKeyPts = 4
             
            while(numberOfKeyPts <= 6):
                keyPts, lineThickness = ShapePolygonApproximation.getKeyPoints(cnt, numberOfKeyPts)
                # print("keyPts = ", keyPts)
                c = np.array(keyPts)
                print(c)
                subvect = []
                for k in range(numberOfKeyPts):
                    q = (k+1) % len(keyPts)
                    p = (k+2) % len(keyPts)
                    a = int(getClockwiseAngle(keyPts[k][0], keyPts[k][1], keyPts[q][0], keyPts[q][1], keyPts[p][0], keyPts[p][1]))
                    subvect.append(a)
                    print(a)
                
                subvect.sort()
                  
                print(subvect) 
                for featurePt in subvect:
                    data.append(featurePt)
                numberOfKeyPts += 2 
    
    #             img1 = img.copy()
    #             # cv2.drawContours(img1, [c], -1, (0, 255, 0), (2 * lineThickness))
    #             cv2.drawContours(img1, [c], -1, (0, 255, 0), 5)
    #             opacity = 0.6
    #             cv2.addWeighted(img1, opacity, img2, 1 - opacity, 0, img2)
    #             cv2.imshow(dataset+"::"+str(classMemberCnt), img2)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
             
                     
            data.append(dataset)
    #         for valueIndex in range(0, len(data)):
    #             if valueIndex == len(data) - 1:
    #                 print(data[valueIndex])
    #             else:
    #                 print(data[valueIndex], end=",")
            with open('dataset.csv', 'a') as csvFile:
                writer = csv.writer(csvFile, delimiter=',')
                #writer = csv.writer(csvFile)
                writer.writerow(data)         
            csvFile.close()
if __name__ == "__main__":
    main()        