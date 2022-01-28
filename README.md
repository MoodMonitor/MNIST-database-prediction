# MNIST-database-prediction

## Goal
Create a machine learning model, which will predict my handwritten digits



## Summary results
The whole presentation of results is in file: "Prediction_results_presentation.ipynb"
### First model, train set: MNIST database, validation data: 110 images from /Digits/


Comparison of datasets:


| MNIST data | <img src="https://user-images.githubusercontent.com/81371889/151608397-ed8045bd-c1a5-4bb7-83d7-8a7ac6ea14ab.png" width="112" height="112"> | <img src="https://user-images.githubusercontent.com/81371889/151608403-b3931da3-b9ec-43c9-8786-bf5a0835af7f.png" width="112" height="112"> |<img src="https://user-images.githubusercontent.com/81371889/151608410-c6e64553-6743-4810-8a41-fd27b3b619c5.png" width="112" height="112">| <img src="https://user-images.githubusercontent.com/81371889/151608416-feef1577-db0e-48a8-8084-c777264bc808.png" width="112" height="112">|
| :---: |:---:| :---:|:---: | :---: |
| **My data** | <img src="https://user-images.githubusercontent.com/81371889/151614590-f5d81686-bb9a-4db8-8a04-af9ad068638b.png" width="112" height="112"> | <img src="https://user-images.githubusercontent.com/81371889/151616527-7163145a-939c-414b-b1ea-5c76b182fdf9.png" width="112" height="112"> |<img src="https://user-images.githubusercontent.com/81371889/151614577-b46eab1a-021b-4726-a7cf-c89d0bdb77a7.png" width="112" height="112">| <img src="https://user-images.githubusercontent.com/81371889/151614563-6bba8a3d-21cb-4ef6-8446-d7997633dabe.png" width="112" height="112">| | | | 

The accuracy I got:  0.5909090638160706

After computed padding via brute force, the accuracy increases to 0.9909090995788574

| **Data after padding** | <img src="https://user-images.githubusercontent.com/81371889/151616653-3632620e-feb5-4a2f-9a8d-04384287b21a.png" width="112" height="112"> | <img src="https://user-images.githubusercontent.com/81371889/151614487-dd20c20f-4035-42ec-8c42-a05a050b1098.png" width="112" height="112"> |<img src="https://user-images.githubusercontent.com/81371889/151616564-0ff4f8fd-3283-4f8c-8027-d18a0546b90b.png" width="112" height="112">| <img src="https://user-images.githubusercontent.com/81371889/151616538-0a5f710e-f9fb-4bc6-8574-afa7ad239be2.png" width="112" height="112">|
| :---: |:---:| :---:|:---: | :---: |

### Second model, train set: MNIST database, validation data: 110 images from /Digits/ with changed shape from rectangle to square
| **Squared data** | <img src="https://user-images.githubusercontent.com/81371889/151614521-eee907ca-3f1f-45a7-a953-fdd2722ee5b1.png" width="112" height="112"> | <img src="https://user-images.githubusercontent.com/81371889/151614487-dd20c20f-4035-42ec-8c42-a05a050b1098.png" width="112" height="112"> |<img src="https://user-images.githubusercontent.com/81371889/151614507-4d8b16b2-aa28-4098-a614-222a869e47be.png" width="112" height="112">| <img src="https://user-images.githubusercontent.com/81371889/151614496-cd40c637-1d61-4885-ade4-04262e282e8e.png" width="112" height="112">|
| :---: |:---:| :---:|:---: | :---: |

The accuracy I got: 0.8181818127632141

After computed padding via brute force, the accuracy increases to 1.00

| **Data after padding** | <img src="https://user-images.githubusercontent.com/81371889/151617419-7203409b-6fe9-4e76-96da-82df78a94a96.png" width="112" height="112"> | <img src="https://user-images.githubusercontent.com/81371889/151617383-53d68a07-2eb1-47a1-952f-993d3c9afd05.png" width="112" height="112"> |<img src="https://user-images.githubusercontent.com/81371889/151617408-4f7ab0ad-460b-42d6-97eb-d796ffd0554b.png" width="112" height="112">| <img src="https://user-images.githubusercontent.com/81371889/151617392-6489201f-05a9-481d-8c96-c31d74ab780e.png" width="112" height="112">|
| :---: |:---:| :---:|:---: | :---: |


## GUI
The GUI allows to create data and make a predictions

<p float="left">
  <img src="https://user-images.githubusercontent.com/81371889/151619820-080f9f4c-d95d-4d15-8f19-52fc0868ca95.png" width="400" />
  <img src="https://user-images.githubusercontent.com/81371889/151619821-602e4c55-9aeb-4c06-be5d-2c1ada8b1e78.png" width="400" /> 
</p>
