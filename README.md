# Deep neural networks models comparison for intestines scans segmentation with transfer learning

This folder contains:
- The final report of this project 
- A video of 5min describing the project 
- Python files containing the code for the pre processing of the images, and the training and evaluation of the models.


## Dataset 

The dataset is a public dataset that is available on Kaggle at this link:

https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data

It is composed of around 17'000 images and a csv file containing the pixel lists of organs on the scans. 

## Mask R-CNN code

The code for the pre process, training and visualising the Mask R-CNN model is in the jupyter notebook `DL_Mask_R_CNN.ipynb`.


## Unet, Unet++ and combined model codes
### Pre process 

From the csv file `train.csv` contained in the Kaggle dataset, the mask images can be obtained by executing:
```
python preprocess/create_mask.py
```
All the paths are hard-coded.


### Training 

In order to train one of the three model (unet, unet++ or the combined model), one must for example execute:

```
python train.py --batch_size 16 --epochs 2 --model unet
```
`--model` can take the values 'unet', 'unetpp' or 'combined'. 
Again the paths where to save the models, train losses and validation scores are hard-coded. 


### Evaluation 

The evaluation is done at the end of every epoch in the `train.py` file. The predictions of the models can be visualised using `plot_predictions.py` functions.