# Project Description: Utilizing CICDDOS-2019 Dataset for Deep Learning Model Training and Real-world Application with CICFlowMeter-V3 (IE105-FinalProject)

## Overview
This project, a course project at the University of Information Technology - VNUHCM, aims to develop a deep learning-based system to detect Distributed Denial of Service (DDoS) attacks using the CICDDOS-2019 dataset. The project involves training an Artificial Neural Network (ANN) model and deploying it for real-time DDoS detection using CICFlowMeter-V3.

## Dataset
The [CICDDOS-2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html), created by the Canadian Institute for Cybersecurity, contains comprehensive records of network traffic, including both benign and DDoS attack traffic. It features a variety of attack types, such as UDP Flood, TCP SYN Flood, and ICMP Flood, providing a rich set of data for training and evaluation.

## Tools and Technologies
- **Deep Learning Framework:** TensorFlow/Keras
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Real-time Flow Analysis:** CICFlowMeter-V3

## Project Steps

### Data Preprocessing
1. Load the CICDDOS-2019 dataset.
2. Clean the data by handling missing values and converting categorical values to numerical.
3. Normalize the feature values to ensure efficient model training.
4. Split the data into training and testing sets.

### Feature Engineering
1. Select relevant features that contribute most to distinguishing between benign and malicious traffic.
2. Use domain knowledge and statistical methods to enhance feature selection.

### Model Development
1. Define the architecture of the ANN model using Keras. The model consists of:
    - An input layer corresponding to the number of features.
    - Two hidden layers with ReLU activation functions.
    - A dropout layer to prevent overfitting.
    - An output layer with a sigmoid activation function for binary classification (benign or DDoS).
2. Compile the model with the Adam optimizer and binary cross-entropy loss function.
3. Train the model on the training data and validate it using the testing data.

### Evaluation
1. Assess the model's performance using metrics such as accuracy, precision, recall, and F1-score.
2. Plot the training and validation loss/accuracy to visualize the model's learning process and identify potential overfitting or underfitting issues.

### Deployment with CICFlowMeter-V3
1. Integrate the trained model with CICFlowMeter-V3 to enable real-time flow analysis and DDoS detection.
2. CICFlowMeter-V3 captures live network traffic and extracts features similar to those in the CICDDOS-2019 dataset.
3. The extracted features are fed into the ANN model to predict whether the traffic is benign or malicious.
4. Implement a notification or mitigation system to alert administrators or take automatic action in case of a detected DDoS attack.

### Real-world Application
1. Deploy the system in a real-world network environment to monitor traffic continuously.
2. Evaluate the system's performance in detecting live DDoS attacks and reducing false positives/negatives.
3. Continuously update the model with new data to improve its robustness and adaptability to evolving attack patterns.

## Conclusion
This project demonstrates a comprehensive approach to developing a deep learning-based DDoS detection system using the CICDDOS-2019 dataset. By integrating the trained ANN model with CICFlowMeter-V3, the system can provide real-time protection against DDoS attacks, ensuring the security and availability of network resources in a real-world environment.
