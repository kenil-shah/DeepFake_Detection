# DeepFake Detection

Implementation for DeepFake Detection Using Ensembling Techniques

![Alt Text](https://github.com/kenil-shah/DeepFake_Detection/blob/master/extra/arch.png)

**Determining whether a given video is Real or Fake by cropping the face of a person and classifying the
cropped image by ensembling ResNext and EfficientNetB6**

## Installation

1) Clone this repository.
```
https://github.com/kenil-shah/DeepFake_Detection.git
```

2) In the repository, execute `pip install -r requirements.txt` to install all the necessary libraries.

3) Three deep learning models are used inorder to determine the class of video
	1) *YOLO Face model:- Used to determine the coordinates of the face of person and generate a cropped facial image using those coordinates*
	2) *ResNext:- First Model used for ensembling*
	3) *EfficientNetB6(With Attention):- Second Model used for ensembling*

4) Download the pretrained weights.
	1) YOLO Face model [pretrained weights](https://drive.google.com/file/d/1pK6NsrMfUqZGIwl9Nihs_WRZ7ooRsrj0/view?usp=sharing) and save it in /model_data/
	2) ResNext:- [pretrained weights](https://drive.google.com/file/d/1_YBJzBd5NOJ_K0-AJMRYUBpXVMfDf4u5/view?usp=sharing) and save it in the root directory 
	3) EfficientNetB6(With Attention)[pretrained weight](https://drive.google.com/file/d/1I7CJ6avdtQblEBBd9b2qXq_Ec940HA7_/view?usp=sharing) and save it in the root directory
