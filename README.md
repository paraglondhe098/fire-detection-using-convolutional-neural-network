# Fire Detection Model using CNN

## Introduction

This project focuses on detecting the presence of fire in images using Convolutional Neural Networks (CNNs).
The goal of the project is to predict whether a given image contains fire (1) or not (0) based on the visual
content of the image. The model can be used for early fire detection and prevention.

## Dataset

The dataset used in this project consists of images collected from various sources,
including Kaggle. The dataset is labeled with corresponding class labels indicating 
whether each image contains fire or not. The dataset is diverse and contains a range of images with and without fire.

## Requirements

To run the project, you need the following dependencies:

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras (>=2.0)
- NumPy (>=1.18)
- Matplotlib (>=3.2)
- Scikit-learn (>=0.22)

You can install the required libraries using pip with the following command:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

## Project Structure

The project repository is organized as follows:

```
├── data/
│   └── Test_Data/        # Directory containing the dataset images
|        └── Fire/
|        └── Non_Fire/
│   └── Train_Data/
|        └── Fire/
|        └── Non_Fire/
|
├── Test_images/           #  Move images you want to detect fire in this directory
|            
|
├── main.py                #  Main Program ( Run using Python interpreter)        
|
|
├── Detect_in_videos.py                #  Detect fire live using a webcam
|
|
├── fire-detection.ipynb               # Jupyter notebook containing the main code and analysis
│ 
│
├── models/
│   └── {List of models made with date in .keras format}        # Pre-trained CNN model for fire detection
│
|
├── README.md              # Project documentation (you are here)
│
└── requirements.txt       # List of project dependencies
```

## Running the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/fire-detection-cnn.git
cd fire-detection-cnn
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset images and place them in the `data/` directory.

4. Open the `fire-detection.ipynb` notebook using Jupyter Notebook or Jupyter Lab to explore the data, train the CNN model, and perform fire detection on the images.

5. Add images to be tested in the Test_images directory.

6. Run main.py to check the performance of model on any image.

## Model Performance

The CNN model has been trained on the dataset, and the pre-trained model is provided in the `models/` directory.
The latest model achieved an accuracy of 82% on the test set and demonstrated its effectiveness in detecting fire in images.

## Conclusion

This project showcases the implementation of a CNN-based fire detection model for images. The model can be a valuable tool for early fire detection and prevention in various applications. You can further enhance and customize the project by experimenting with different CNN architectures, data augmentation techniques, and training strategies.

Feel free to use, modify, and extend this project for your own applications. If you have any questions or suggestions, please don't hesitate to reach me out.

Stay safe and vigilant against fires!
