from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QDirModel
import numpy as np
from spectral import *
from numpy import linalg as LA
import statistics
import math
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class Ui_MainWindow(QWidget):
    # Load and show an hyperspectral image file
    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Spectral Data', '#FilePath#',"Spectral Image Files (*.lan *.hdr)")
        if fname[0] != "":
            global img, rowNum, colNum, bandNum
            img = open_image(fname[0])
            rowNum = img.nrows
            colNum = img.ncols
            bandNum = img.nbands
            #view_cube(img) //view hypercube
            imshow(img, (29, 19, 9))
            self.lbl_tutorial.setVisible(False)
            self.lbl_info.setVisible(True)
            self.btn_selectedBand.setEnabled(True)
            self.btn_allLambda.setEnabled(True)
            self.btn_lambda.setEnabled(True)
            self.btn_pca2hsi.setEnabled(True)
            self.spinBox_bandNo.setVisible(True)
            self.spinBox_featureNo.setVisible(True)
            self.lbl_bandNumber.setVisible(True)
            self.lbl_lambdaCount.setVisible(True)
            self.lbl_infoText.setText(str(img).lstrip())

    # Show n'th band
    def getSelectedBand(self):
        selectedBandNo = self.spinBox_bandNo.value()
        if selectedBandNo < bandNum : 
            self.lbl_errorText.clear()
            selectedBand = img.read_band(selectedBandNo)
            imshow(selectedBand)
        else : 
            self.lbl_errorText.setText("Invalid band number !")

    # Calculates eigenvalue's
    def pca(self):
        self.listWidget_screen.clear()
        global uMatrix, matrix 
        matrix = img.load()
        uMatrix = np.zeros([rowNum * colNum, bandNum])
        c = 0
        for i in range(rowNum):
            for j in range(colNum):
                uMatrix[c,:] = matrix[i,j,:].reshape(bandNum)
                c = c + 1

        # Normalization
        global normMatrix
        summary = 0
        for i in range(rowNum * colNum):
            summary = summary + (uMatrix[i][0]**2)  
        c = math.sqrt(summary)
        
        normMatrix = uMatrix / c

        transpose_matrix = normMatrix.transpose()
        covariance_matrix = transpose_matrix.dot(normMatrix)

        global eigvec
        eigval, eigvec = LA.eig(covariance_matrix)
        lambdas = []
        for ev in eigval:
            lamb = ev / sum(eigval)
            lambdas.append(lamb)
        return lambdas

    # Calculate all bands eigenvalue's
    def allPca(self):
        lambdas = self.pca()
        i = 1
        for l in lambdas:
            text = 'PC ' + str(i) + ' = ' + str(l)
            self.listWidget_screen.addItem(text)
            i += 1
    
    # Calculates the best n eigenvalue's
    def selectedPca(self):
        lambdas = self.pca()
        featureCount = self.spinBox_featureNo.value()
        if featureCount < len(lambdas) :
            self.lbl_errorText.clear()
            i = 1
            while i <= featureCount:
                text = 'PC ' + str(i) + ' = ' + str(lambdas[i-1])
                self.listWidget_screen.addItem(text)
                i += 1
        else : self.lbl_errorText.setText("Invalid feature number !")

    # Applying PCA to hyperspectral image file
    def pcaToHsi(self):
        lambdas = self.pca()
        PCs_array = np.array([eigvec[0], eigvec[1], eigvec[2]])
        tPCs_array = PCs_array.transpose()
        numOfPCs = tPCs_array.shape[1]
        result_matrix = np.dot(normMatrix, tPCs_array)

        # Folding array to image
        global fMatrix
        fMatrix = np.zeros([rowNum, colNum, numOfPCs])
        c = 0
        for i in range(rowNum):
            for j in range(colNum):
                fMatrix[i,j,:] = result_matrix[c,:].reshape(1,1,numOfPCs)
                c = c + 1

        imshow(fMatrix)

        self.listWidget_screen.addItem('PCA Applied !\nThe top three principal values with high information :\n')
        i = 1
        truthVal = 0
        while i <= 3:
            truthVal += lambdas[i-1]
            text = 'PC ' + str(i) + ' = {:.3}'.format(lambdas[i-1])
            self.listWidget_screen.addItem(text)
            i += 1
        self.listWidget_screen.addItem('Truth Value = {:.5}'.format(truthVal))

        self.btn_slic.setEnabled(True),
        self.lbl_segmentNumber.setVisible(True)
        self.comboBox_segmentNumber.setVisible(True)

    # Applying SLIC
    def applySlic(self):
        segmentNum = int(self.comboBox_segmentNumber.currentText())
        global segments 
        segments = slic(fMatrix, n_segments = segmentNum, compactness = 0.5)
        imshow(mark_boundaries(fMatrix, segments, color=0.01), figsize = (7,7))

        # Finding mean of each regions
        regions = regionprops(segments)
        meanRegionMatrix = []
        for region in regions:
            temp = np.zeros([bandNum])
            for coord in region.coords:
                temp += uMatrix[(len(matrix) * coord[0]) + coord[1]]
            meanRegion = temp / region.area
            meanRegionMatrix.append(meanRegion)

        # Refilling all pixels with their superpixel group's mean
        global meanMatrix
        meanMatrix = uMatrix
        for i in range(len(regions)):
            for coord in regions[i].coords:
                meanMatrix[(len(matrix) * coord[0]) + coord[1]] = meanRegionMatrix[i]
        
        self.btn_meanImage.setEnabled(True)
        self.btn_svm.setEnabled(True)

    # Finding mean image
    def showMeanImage(self):
        # Folding mean image
        meanImage = np.zeros([rowNum, colNum, bandNum])
        c = 0
        for i in range(rowNum):
            for j in range(colNum):
                meanImage[i,j,:] = meanMatrix[c,:].reshape(1,1,bandNum)
                c = c + 1

        imshow(meanImage, figsize=(7,7))

    # Apply SVM
    def applySVM(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Ground Truth Data', '#FilePath#',"Spectral Image Files (*.GIS)")
        if fname[0] != "":
            self.listWidget_screen.clear()
            img_gt = open_image(fname[0]).read_band(0)

            # Finding background pixel coordinates
            backgroundCoords = []
            for i in range(len(img_gt)):
                for j in range(len(img_gt[0])):
                    if img_gt[i][j] == 0:
                        bg = [i, j]
                        backgroundCoords.append(bg)

            # Removing backgrounds on meanMatrix
            backgroundlessMatrix = meanMatrix.copy()
            for coord in backgroundCoords:
                backgroundlessMatrix[(len(matrix) * coord[0]) + coord[1]] = 0

            # Remove backgrounds
            finalMatrix = []
            for pixel in backgroundlessMatrix:
                if pixel.all() != 0:
                    finalMatrix.append(pixel)
            # Normalize backgroundless image
            scaler = StandardScaler()
            normMatrix = scaler.fit_transform(finalMatrix)

            # Finding and removing ground-truth image backgrounds
            gt_nobg = []
            for row in img_gt:
                for px in row:
                    if px != 0:
                        gt_nobg.append(px)

            # Creating model
            model = SVC(C=1, gamma=1, kernel='rbf')

            text = 'The SVM model is run 10 times on shuffled data and finally\nthe mean score and standard deviation is calculated.'
            self.listWidget_screen.addItem(text)

            # Fitting and testing model
            scores = []
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(normMatrix, gt_nobg, test_size=0.9)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                text = 'Run ' + str(i+1) + ': {:.3}'.format(score)
                self.listWidget_screen.addItem(text)
                scores.append(score)

            self.listWidget_screen.addItem('Mean Score: {:.3}'.format(statistics.mean(scores)))
            self.listWidget_screen.addItem('Standard Deviation: {:.3}'.format(statistics.stdev(scores)))
            
    ################
    # Designing UI #
    ################
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(850, 650)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Fonts
        f_tahoma16 = QtGui.QFont()
        f_tahoma16.setFamily("Tahoma")
        f_tahoma16.setPointSize(16)

        f_tahoma9 = QtGui.QFont()
        f_tahoma9.setFamily("Tahoma")
        f_tahoma9.setPointSize(9)
        
        f_tahoma12 = QtGui.QFont()
        f_tahoma12.setFamily("Tahoma")
        f_tahoma12.setPointSize(12)
        
        f_tahoma14 = QtGui.QFont()
        f_tahoma14.setFamily("Tahoma")
        f_tahoma14.setPointSize(14)

        f_tahoma11 = QtGui.QFont()
        f_tahoma11.setFamily("Tahoma")
        f_tahoma11.setPointSize(11)

        # Labels
        self.lbl_errorText = QtWidgets.QLabel(self.centralwidget)
        self.lbl_errorText.setGeometry(QtCore.QRect(50, 210, 250, 50))
        self.lbl_errorText.setFont(f_tahoma14)
        self.lbl_errorText.setStyleSheet("color: red")
        self.lbl_errorText.setObjectName("lbl_errorText")

        self.lbl_lambdaCount = QtWidgets.QLabel(self.centralwidget)
        self.lbl_lambdaCount.setGeometry(QtCore.QRect(160, 360, 71, 51))
        self.lbl_lambdaCount.setFont(f_tahoma14)
        self.lbl_lambdaCount.setVisible(False)
        self.lbl_lambdaCount.setObjectName("lbl_lambdaCount")

        self.lbl_segmentNumber = QtWidgets.QLabel(self.centralwidget)
        self.lbl_segmentNumber.setGeometry(QtCore.QRect(160, 500, 71, 51))
        self.lbl_segmentNumber.setFont(f_tahoma14)
        self.lbl_segmentNumber.setVisible(False)
        self.lbl_segmentNumber.setObjectName("lbl_segmentNumber")

        self.lbl_title = QtWidgets.QLabel(self.centralwidget)
        self.lbl_title.setGeometry(QtCore.QRect(250, 10, 381, 31))
        self.lbl_title.setFont(f_tahoma16)
        self.lbl_title.setObjectName("lbl_title")
        
        self.lbl_info = QtWidgets.QLabel(self.centralwidget)
        self.lbl_info.setGeometry(QtCore.QRect(330, 60, 181, 31))
        self.lbl_info.setFont(f_tahoma12)
        self.lbl_info.setVisible(False)
        self.lbl_info.setObjectName("lbl_info")
        
        self.lbl_bandNumber = QtWidgets.QLabel(self.centralwidget)
        self.lbl_bandNumber.setGeometry(QtCore.QRect(160, 130, 71, 51))
        self.lbl_bandNumber.setFont(f_tahoma14)
        self.lbl_bandNumber.setVisible(False)
        self.lbl_bandNumber.setObjectName("lbl_bandNumber")
        
        self.lbl_infoText = QtWidgets.QLabel(self.centralwidget)
        self.lbl_infoText.setGeometry(QtCore.QRect(340, 90, 490, 180))
        self.lbl_infoText.setFont(f_tahoma11)
        self.lbl_infoText.setText("")
        self.lbl_infoText.setWordWrap(True)
        self.lbl_infoText.setObjectName("lbl_infoText")

        self.lbl_tutorial = QtWidgets.QLabel(self.centralwidget)
        self.lbl_tutorial.setGeometry(QtCore.QRect(290, 75, 540, 180))
        self.lbl_tutorial.setFont(f_tahoma9)
        self.lbl_tutorial.setObjectName("lbl_tutorial")
        
        # Buttons
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(30, 60, 241, 51))
        self.btn_load.setFont(f_tahoma12)
        self.btn_load.clicked.connect(self.loadImage)
        self.btn_load.setObjectName("btn_load")

        self.btn_allLambda = QtWidgets.QPushButton(self.centralwidget)
        self.btn_allLambda.setEnabled(False)
        self.btn_allLambda.setGeometry(QtCore.QRect(30, 290, 135, 50))
        self.btn_allLambda.setFont(f_tahoma12)
        self.btn_allLambda.clicked.connect(self.allPca)
        self.btn_allLambda.setObjectName("btn_allLambda")

        self.btn_lambda = QtWidgets.QPushButton(self.centralwidget)
        self.btn_lambda.setEnabled(False)
        self.btn_lambda.setGeometry(QtCore.QRect(30, 360, 121, 51))
        self.btn_lambda.setFont(f_tahoma12)
        self.btn_lambda.clicked.connect(self.selectedPca)
        self.btn_lambda.setObjectName("btn_lambda")
        
        self.btn_pca2hsi = QtWidgets.QPushButton(self.centralwidget)
        self.btn_pca2hsi.setEnabled(False)
        self.btn_pca2hsi.setGeometry(QtCore.QRect(30, 430, 121, 51))
        self.btn_pca2hsi.setFont(f_tahoma12)
        self.btn_pca2hsi.clicked.connect(self.pcaToHsi)
        self.btn_pca2hsi.setObjectName("btn_pca2hsi")

        self.btn_slic = QtWidgets.QPushButton(self.centralwidget)
        self.btn_slic.setEnabled(False)
        self.btn_slic.setGeometry(QtCore.QRect(30, 500, 121, 51))
        self.btn_slic.setFont(f_tahoma12)
        self.btn_slic.clicked.connect(self.applySlic)
        self.btn_slic.setObjectName("btn_slic")
        
        self.btn_selectedBand = QtWidgets.QPushButton(self.centralwidget)
        self.btn_selectedBand.setEnabled(False)
        self.btn_selectedBand.setGeometry(QtCore.QRect(30, 130, 121, 51))
        self.btn_selectedBand.setFont(f_tahoma12)
        self.btn_selectedBand.clicked.connect(self.getSelectedBand)
        self.btn_selectedBand.setObjectName("btn_selectedBand")

        self.btn_meanImage = QtWidgets.QPushButton(self.centralwidget)
        self.btn_meanImage.setEnabled(False)
        self.btn_meanImage.setGeometry(QtCore.QRect(30, 570, 121, 51))
        self.btn_meanImage.setFont(f_tahoma12)
        self.btn_meanImage.clicked.connect(self.showMeanImage)
        self.btn_meanImage.setObjectName("btn_meanImage")

        self.btn_svm = QtWidgets.QPushButton(self.centralwidget)
        self.btn_svm.setEnabled(False)
        self.btn_svm.setGeometry(QtCore.QRect(190, 570, 121, 51))
        self.btn_svm.setFont(f_tahoma12)
        self.btn_svm.clicked.connect(self.applySVM)
        self.btn_svm.setObjectName("btn_svm")
        
        # Spin Boxes
        self.spinBox_featureNo = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_featureNo.setGeometry(QtCore.QRect(240, 370, 71, 31))
        self.spinBox_featureNo.setFont(f_tahoma11)
        self.spinBox_featureNo.setMaximum(999)
        self.spinBox_featureNo.setVisible(False)
        self.spinBox_featureNo.setObjectName("spinBox_featureNo")
        
        self.spinBox_bandNo = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_bandNo.setGeometry(QtCore.QRect(240, 140, 71, 31))
        self.spinBox_bandNo.setFont(f_tahoma11)
        self.spinBox_bandNo.setMaximum(999)
        self.spinBox_bandNo.setVisible(False)
        self.spinBox_bandNo.setObjectName("spinBox_bandNo")

        # List Widget
        self.listWidget_screen = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_screen.setGeometry(QtCore.QRect(340, 280, 490, 340))
        self.listWidget_screen.setFont(f_tahoma11)
        self.listWidget_screen.setObjectName("listWidget_screen")

        # Combo Box
        self.comboBox_segmentNumber = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_segmentNumber.setGeometry(QtCore.QRect(240, 510, 71, 31))
        self.comboBox_segmentNumber.setFont(f_tahoma11)
        self.comboBox_segmentNumber.addItem("100")
        self.comboBox_segmentNumber.addItem("200")
        self.comboBox_segmentNumber.addItem("300")
        self.comboBox_segmentNumber.addItem("400")
        self.comboBox_segmentNumber.addItem("500")
        self.comboBox_segmentNumber.setVisible(False)
        self.comboBox_segmentNumber.setObjectName("comboBox_segmentNumber")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 850, 26))
        self.menubar.setObjectName("menubar")
        self.menuSpectralPY = QtWidgets.QMenu(self.menubar)
        self.menuSpectralPY.setObjectName("menuSpectralPY")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuSpectralPY.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HSI Analyser"))
        self.btn_load.setText(_translate("MainWindow", "LOAD IMAGE"))
        self.lbl_lambdaCount.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Lambda <br/>Count</span></p></body></html>"))
        self.lbl_segmentNumber.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Segment <br/>Number</span></p></body></html>"))
        self.btn_lambda.setText(_translate("MainWindow", "LAMBDA"))
        self.btn_allLambda.setText(_translate("MainWindow", "ALL LAMBDAS"))
        self.lbl_title.setText(_translate("MainWindow", "Hyperspectral Image Analyser"))
        self.btn_pca2hsi.setText(_translate("MainWindow", "PCA2HSI"))
        self.btn_slic.setText(_translate("MainWindow", "SLIC"))
        self.btn_svm.setText(_translate("MainWindow", "SVM"))
        self.btn_meanImage.setText(_translate("MainWindow", "MEAN IMAGE"))
        self.lbl_info.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Image Information</p></body></html>"))
        self.lbl_bandNumber.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Band<br/>Number</span></p></body></html>"))
        self.btn_selectedBand.setText(_translate("MainWindow", "BAND"))
        self.lbl_tutorial.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt;\">Welcome to Hyperspectral Image Analyser !</span></p><p><span style=\" font-size:9pt;\">- Click the 'LOAD IMAGE' button to open an hyperspectral image<br/>- Enter a band number and click 'BAND' button to show a band from image<br/>- Click 'ALL LAMBDAS' button to calculate all lambdas<br/>- To calculate first n lambdas, enter a Feature Number and click 'LAMBDA' button<br/>- To apply PCA on the hyperspectral image, click 'PCA2HSI' button<br/>- To find and see superpixels, click 'SLIC' button<br/>- To Apply SVM on dataset and calculate predict scores, click 'SVM' button and<br/>  choose ground truth image file. </span></p></body></html>"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
