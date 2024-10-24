# Biosignals Time Series Dataset

This repository aims to present the development of research in computer vision and information processing applied to the detection of human biosignals. As a first milestone, the proposal, and development of a dataset are presented to enable the training of neural networks from signal signatures in time series.

## Research Objective

To perform the extraction, analysis, and interpretation of human biosignals using a technological route that standardizes visual acquisition via video cameras in normalized measures that describe behavior signatures from multivariate time series.

## Introduction

In the context of video image processing, the conventional technological route is presented in Figure 1 by flow 1. This approach is the most used because it focuses on multilayer detection and the use of neural networks in the image domain. On the other hand, this research addresses another route, 2, which creates computational structures that extract and normalize the information of interest and perform analyses in the domain of time series.

<div align="center">
  <img src="https://github.com/jomas007/biosignals-time-series-dataset/blob/main/images/README_imgs/rect7.png" alt="souza94" width="400"/>
  <p>Figure 1 - Approach 1 deals with neural networks in the image domain and 2 focuses on neural network processing in the domain of multivariate time series.</p>
</div>

## Proposal of computational system

In general terms, the ideal computational solution in the approach presented above would have the following functionalities as listed below, enumerated according to Figure 2:

1. Detect and crop the human face in the image, separating it from other elements in the scene.
2. Identify the points of interest that describe the face.
3. Calibrate, normalize and standardize observations to generate measurable and comparable criteria.
4. Extract measurements from spatial and temporal descriptors, and generate multivariate time series.
5. Perform analyses, cross-links, and correlations with real-world events.

<div align="center">
  <img src="https://github.com/jomas007/biosignals-time-series-dataset/blob/main/images/README_imgs/rect11.png" alt="souza1" width="500"/>
  <p>Figure 2 - Functionalities and requirements of the proposed computational system.</p>
</div>

## Dataset Generation Flow

The first principle for this previously mentioned proposal to become feasible is the availability of time series data extracted from videos with diverse conditions. Thus, it is necessary to have organized datasets that represent various classes of possibilities that enable the training of the neural networks necessary to meet the functionalities presented in Figure 2.

In Figure 3 below, the main modules comprising the two stages of the project (A and B) are indicated.

<div align="center">
  <img src="https://github.com/jomas007/biosignals-time-series-dataset/blob/main/images/README_imgs/rect13.png" alt="souza2" width="650"/>
  <p>Figure 3 - Dataset Stages, A represents the raw data generation and B the Labeling and Qualification flow.</p>
</div>

For a more detailed description of each block:

- [Data generation flow](https://github.com/jomas007/biosignals-time-series-dataset/wiki/Blocks-Description)

## How to install

- [Installation guide for **Windows**](https://github.com/jomas007/biosignals-time-series-dataset/wiki/How-to-install#windows)
- [Installation guide for **Linux**](https://github.com/jomas007/biosignals-time-series-dataset/wiki/How-to-install#linux)

## How to run the application

If you want to **just run the application** and see how it works:

- [Run the application guide](https://github.com/jomas007/biosignals-time-series-dataset/wiki/How-to-use#Run-the-application)

If you want to **use your own** source of videos:

- [Do it your self guide](https://github.com/jomas007/biosignals-time-series-dataset/wiki/How-to-use#Do-it-your-self)

## Future Developments

- Create a better normalization process, for the Yaw and Pitch head movements to be normalized.
- Improve the existent neural network for a better performance.
