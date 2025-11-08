# Pothole Detection with YOLOv8

## View Project: [pothole-detect.streamlit.app](https://pothole-detect.streamlit.app/)

This repository contains the complete workflow for training a YOLOv8s object detection model to identify potholes in road images. The project covers the full pipeline: sourcing raw images, performing manual data labeling, and training the model.

## Project Overview

The goal of this project was to learn the end-to-end process of building a custom object detector. The model is trained on a single class: `pothole`.



## Workflow and Tools

This project was built using the following components:

* **Model:** **YOLOv8s** (small variant) by Ultralytics.
* **Data Sourcing:** Raw images were taken from the [Pothole Image Segmentation Dataset](https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset) on Kaggle.
* **Data Labeling:** Manual bounding box annotation was performed using **Label Studio**. The labels were then exported in YOLO `.txt` format.
* **Training:** The model was trained using a free GPU on **Google Colab**. The notebook used is based on the [Train YOLO Models](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb) template.

## How to Use

The main training process is documented in the included Jupyter Notebook. To run the training, you will need to:

1.  Have your own labeled dataset (images and YOLO-formatted labels) uploaded to a service like Google Drive.
2.  Create a `data.yaml` file that points to your training and validation data paths.
3.  Run the cells in the notebook to install dependencies, mount your drive, and begin the training.
