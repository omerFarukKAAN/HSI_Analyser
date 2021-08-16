# Hyperspectral Image Analyser
This tool can be help hyperspectral image analysis and calculations.

### With this tool, you can:
- *Show hyperspectral image data*
- *Show n'th band of image data* 
- *Calculate all eigenvalues or first n eigenvalues and truth value*
- *Apply 'Principal Component Analysis' to data*
- *Apply SLIC algorithm with selected segment count and show segment areas*
- *Show mean image of each segment area after SLIC*
- *Finally, apply 'Support Vector Machines' algorithm and run 10 times on shuffled data*
- *Show mean score and standart deviation of model predicts*

## *Starting..*
Open cmd and run command for execute hsi_analyser. 
```
python ./hsi_analyser.py
```
Read tutorial text and go step by step
- Click 'LOAD IMAGE' button and select hyperspectral data file (ex. 92AV3C.lan)
- Click 'SVM' button and select a ground truth image for teach SVM Model (ex. 92AV3GT.GIS)
