import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize 
from PyQt5.Qt import QPushButton, QComboBox, QTextEdit, QScrollArea
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os,cv2
import numpy as np
import ShapeFeatureVectorGen
import IEM_Custom_KNN_Classifier

class MyForm(QMainWindow):
    
    def __init__(self):
        self.getTrainingData()
        self.imageWidth = 480
        self.imageHeight = 360
        self.numRetrieval = 9
        QMainWindow.__init__(self)
        
        self.setMinimumSize(QSize(self.imageWidth + 60, int(self.imageHeight * 1.9)))    
        self.setWindowTitle("ShapeRetrievalEngine") 
        centralWidget = QWidget(self) 
        self.queryImageFileName = None
        
         
        imageWidget=QWidget(centralWidget) 
        imageWidget.setGeometry(30,30,self.imageWidth ,self.imageHeight)
        #imageWidget.setAutoFillBackground(True)
        imageWidget.setStyleSheet("background-color:rgb(0,0,0)")
        self.queryImageLabel = QLabel()
        imageWidgetLayout = QVBoxLayout(imageWidget)
        self.queryImageLabel.setAlignment(Qt.AlignCenter)
        self.queryImageLabel.setText("<font color='white'>Query Image</font>")
        imageWidgetLayout.addWidget(self.queryImageLabel)
        
        
        browseButton=QPushButton(centralWidget)
        browseButton.setGeometry(30, self.imageHeight + 40, 60, 30)
        browseButton.setText("Browse")
        browseButton.clicked.connect(self.getfile)
        
        searchButton=QPushButton(centralWidget)
        searchButton.setGeometry(90, self.imageHeight + 40, 60, 30)
        searchButton.setText("Search")
        searchButton.clicked.connect(self.getResultSet)
        
        scrollAreaImageWidgetContents = QWidget(centralWidget)
        scrollAreaImageWidgetContents.setGeometry(20, self.imageHeight  + 90, self.imageWidth + 10, int(self.imageHeight * 0.25) + 90)

        scrollAreaImageWidget = QtWidgets.QGroupBox('Retrieval Results')
        reslayout = QtWidgets.QHBoxLayout()
        self.resultImageLabelList = []
        x = 10
        y = 10
        
        for i in range(self.numRetrieval):
            resSingleLayout = QtWidgets.QVBoxLayout()
            resultWidget = QWidget()
            resultWidget.setGeometry(x, y, int(self.imageWidth * 0.25), int(self.imageHeight * 0.25))
            resultWidget.setFixedSize(QSize(int(self.imageWidth * 0.25), int(self.imageHeight * 0.25)))
            resultWidget.setStyleSheet("background-color:rgb(0,0,0)")
            imageLabel = QLabel()
            imageLabelLayout = QVBoxLayout(resultWidget)
            imageLabel.setAlignment(Qt.AlignCenter)
            imageLabel.setText("<font color='white'>None</font>")
            imageLabelLayout.addWidget(imageLabel)

            x = x + int(self.imageWidth * 0.25) + 10
            resSingleLayout.addWidget(resultWidget)
            resLabel = QLabel()
            resLabel.setAlignment(Qt.AlignCenter)
            resLabel.setText("Image_"+str(i+1))
            resSingleLayout.addWidget(resLabel)
            reslayout.addItem(resSingleLayout)
            self.resultImageLabelList.append(imageLabel)
        
        scrollAreaImageWidget.setLayout(reslayout)
        scrollAreaImage=QScrollArea()
        scrollAreaImage.setWidget(scrollAreaImageWidget)
        scrollAreaImage.setWidgetResizable(False)
        scrollAreaImage.setFixedHeight(int(self.imageHeight * 0.25) + 80)
        scrollAreaImage.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)        
        layout = QVBoxLayout(scrollAreaImageWidgetContents)
        layout.addWidget(scrollAreaImage)

        self.setCentralWidget(centralWidget)
    
    def getTrainingData(self):
        df = pd.read_csv("./dataset.csv")
        self.className = ['apple', 'bell', 'bird', 'frog', 'guitar']
        IEM_Custom_KNN_Classifier.mod_data(df, self.className)
        
        dataset = df.values.tolist()
        #The keys of the dict are the classes that the data is classified into
        self.training_set = {0: [], 1:[], 2:[], 3:[],4:[]}
    
        training_data = dataset[0:len(dataset)]
         
        #Insert data into the training set
        for record in training_data:
            self.training_set[record[-1]].append(record[:-1]) # Append the list in the dict will all the elements of the record except the class

    def getfile(self):
        #queryImageLabel.setText("Hello World")
        #print("Hello World")
        self.queryImageFileName = QFileDialog.getOpenFileName(self, self.tr("Open file"), 
                QDir.currentPath(),self.tr("Image files (*.jpg *.bmp *.png)"), options=QFileDialog.DontUseNativeDialog)
        
        image = QImage(self.queryImageFileName[0])
        image = image.scaled(self.imageWidth - 30, self.imageHeight - 30)
        self.queryImageLabel.setPixmap(QPixmap.fromImage(image))
        self.getRefresh()
    def getRefresh(self):
        for i in range(9):
            self.resultImageLabelList[i].setText("<font color='white'>None</font>")
    def getResultSet(self):
        if self.queryImageFileName == None:
            return
        print('findResultSet')
        s = time.time()
        featureVector = ShapeFeatureVectorGen.getFeatureVector(self.queryImageFileName[0])
        print("featureVector = ", featureVector)
        predicted_class, confidence = IEM_Custom_KNN_Classifier.customKNN_predict(self.training_set, featureVector, 9, len(self.className))
        print("Predicted Class = ", self.className[predicted_class], ", confidence = ", confidence)
        # Define data path
        data_path = './data'
#         print(data_path)
#         data_dir_list = os.listdir(data_path)
#         print(data_dir_list)
        imageCount = 0
#         queryImageFileName = str(QFileInfo(self.queryImageFileName[0]).fileName())
#         queryImageFileName = queryImageFileName.lower()
#         print(queryImageFileName)
        dataset = self.className[predicted_class]
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            #print(data_path + '/'+ dataset + '/'+ img)
            imageCount = imageCount + 1
            if imageCount > self.numRetrieval:
                break
            else:
                resImagePath = data_path + '/'+ dataset + '/'+ img
                #resFilename = QFileInfo(resImagePath).fileName()
                #print(resFilename)
                image = QImage(resImagePath)
                image = image.scaled(int(self.imageWidth * 0.25) - 30, int(self.imageHeight * 0.25) - 30)
                self.resultImageLabelList[imageCount - 1].setPixmap(QPixmap.fromImage(image))
                        
 
        e = time.time()
        print("Exec Time:" ,e-s, " sec")

        print('findResultSet Ends')
import math
import pandas as pd
import random
from sklearn import preprocessing
import time
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MyForm()
    form.show()
    sys.exit( app.exec_() )   
