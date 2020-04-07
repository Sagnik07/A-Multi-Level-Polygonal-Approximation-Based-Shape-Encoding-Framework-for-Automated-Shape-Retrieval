import numpy as np
import math
import cv2



def getLineDistance(p0, p1, p):
    d = 0
    x0 = p0[0]
    y0 = p0[1]
    x1 = p1[0]
    y1 = p1[1]
    x = p[0]
    y = p[1]
    
    if(x0 == x1):
        d = abs(x - x0);
    elif (y0 == y1):
        d = abs(y - y0);
    else:
        d = abs(((y0 - y1) * x + (x1 - x0) * y + (x0 * y1 - x1 * y0))/math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)));
    
    return d;

def getThickLine(hull):
    n = len(hull)
    #0:side, 1:remotest vertex from the side, 2:distance
    thickLineInfo = [n - 1, 0, 0]
    if(n < 3):
        return thickLineInfo
    delmax = 0
    for i in range(1, n - 1):
        d = getLineDistance(hull[n-1], hull[0], hull[i])
        if(delmax < d):
            delmax = d
            k = i
    
    thickLineInfo[1] = k
    thickLineInfo[2] = delmax
    delmin = delmax
    i = 1;
    while (i <= n - 1):
        del1 = getLineDistance(hull[i-1], hull[i], hull[k])
        del2 = getLineDistance(hull[i-1], hull[i], hull[(k+1)%n])
        if(del1 < del2):
            k = (k+1) % n
        else:
            if(del1 < delmin):
                delmin = del1
                thickLineInfo[0] = i - 1
                thickLineInfo[1] = k
                thickLineInfo[2] = delmin
            i = i + 1

    return thickLineInfo        

def getKeyPoints(contour, DP):
    # print("contour = ", contour)
    n = len(contour)
    # print("length of contour = ", n)
    s = 0
    e = n - 1
    keySegments = [[s, e]]
    keySegmentLineThickness = []
    numSeg = 1

    while numSeg  < DP:
        maxCost = 0
        selectedCurveSegmentHull = []
        for i in range(len(keySegments)):
            curveSegment = contour[keySegments[i][0]:keySegments[i][1]]
            # print("curveSegment = ", curveSegment)
            hull = cv2.convexHull(np.array(curveSegment))
            hull = hull.reshape(len(hull), 2).tolist()
            ## print("hull = ", hull)
            ## print("length of hull = ", len(hull))
            thickLine = getThickLine(hull)
            cost = thickLine[2]
            if(len(keySegmentLineThickness) == 0):
                keySegmentLineThickness.append(cost)
            if cost > maxCost:
                maxCost = cost
                s = keySegments[i][0]
                e = keySegments[i][1]
                selectedCurveSegmentHull = hull
        
        print("maxCost = ", maxCost)
        if math.ceil(maxCost) < 4:
            break
        splitMinCost = -1
        selectedPivotPointIndex = -1
        ## print("selectedCurveSegmentHull = ", selectedCurveSegmentHull)
        for i in range(len(selectedCurveSegmentHull)):
            pivotPoint = selectedCurveSegmentHull[i]
            if contour[s] != pivotPoint and contour[e] != pivotPoint:
                # print("pivotPoint = ", pivotPoint)
                pivotPointIndex = contour.index(pivotPoint)
                # print("pivotPointIndex = ", pivotPointIndex)
                curveSegment1 = contour[s:pivotPointIndex]
                # print("curveSegment1 = ", curveSegment1)
                hull1 = cv2.convexHull(np.array(curveSegment1))
                hull1 = hull1.reshape(len(hull1), 2).tolist()
                ## print("hull1 = ", hull1)
                ## print("length of hull1 = ", len(hull1))
                thickLine1 = getThickLine(hull1)
                cost1 = thickLine1[2]

                curveSegment2 = contour[pivotPointIndex:e+1]
                # print("curveSegment2 = ", curveSegment2)
                hull2 = cv2.convexHull(np.array(curveSegment2))
                hull2 = hull2.reshape(len(hull2), 2).tolist()
                ## print("hull2 = ", hull2)
                ## print("length of hull2 = ", len(hull2))
                thickLine2 = getThickLine(hull2)
                cost2 = thickLine2[2]
                
                splitCost = (cost1 + cost2) * abs(cost1 - cost2)
                # print("splitCost = ", splitCost)
                if(splitMinCost == -1 or splitMinCost > splitCost):
                    splitMinCost = splitCost
                    selectedPivotPointIndex = pivotPointIndex
                    selectedCost1 = cost1
                    selectedCost2 = cost2
        removeIndex = keySegments.index([s, e])
        del keySegments[removeIndex]
        del keySegmentLineThickness[removeIndex]
        #keySegments.remove([s, e])
        keySegments.append([s, selectedPivotPointIndex])
        keySegmentLineThickness.append(selectedCost1)
        keySegments.append([selectedPivotPointIndex + 1, e])
        keySegmentLineThickness.append(selectedCost2)
        numSeg = numSeg + 1
        # print("keySegments = ", keySegments)  
        # print("keySegmentLineThickness = ", keySegmentLineThickness) 
        # print("numSeg = ", numSeg, "\n")          

    keyPoints = []
    keyPtContourIndex = []
    for i in range(len(keySegments)):
        keyPtContourIndex.append(keySegments[i][0])
    keyPtContourIndex.sort()
    for i in range(len(keySegments)):
        keyPoints.append(contour[keyPtContourIndex[i]])
    
    maxPolyLineThickness = math.ceil(max(keySegmentLineThickness))
    print("numSeg = ", numSeg)
    print("keySegments = ", keySegments)  
    print("keySegmentLineThickness = ", keySegmentLineThickness) 
    print("keyPoints = ", keyPoints) 
    print("maxPolyLineThickness = ", maxPolyLineThickness)

    return keyPoints, maxPolyLineThickness
def main():            
    top, bottom = 100//2, 100//2
    left, right = 100//2, 100//2
    img = cv2.imread('./data/frog/frog-8.png')
    numberOfKeyPts = 6
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    
    img2 = img.copy();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(thresh_gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    # print("Shape of contour array: ", cnt.shape) 
    cnt = np.array(cnt)
    cnt = cnt.reshape(len(cnt), 2).tolist()
    keyPts, lineThickness = getKeyPoints(cnt, numberOfKeyPts)
    # print("keyPts = ", keyPts)
    c = np.array(keyPts)
    img1 = img.copy()
    # cv2.drawContours(img1, [c], -1, (0, 255, 0), (2 * lineThickness))
    cv2.drawContours(img1, [c], -1, (0, 255, 0), 5)
    opacity = 0.6
    cv2.addWeighted(img1, opacity, img2, 1 - opacity, 0, img2)
    cv2.imshow('PolygonApproximation', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()    