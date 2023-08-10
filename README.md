# Self-Supervised Object Detection on Custom Dataset with SimCLR and SimSiam

#### **Project Summary **
This project utilizes self-supervised learning for vehicle-type detection, the subset of data was taken from ImageNet. However, this code can be implemented on any other objects depending on your dataset.

Pretraining and evaluation is done on one dataset, and transfer learning is performed on another dataset. The closer these 2 datasets to each other, the better results and generalization we will get. In this my case, pretraining was done on 5 types of vehicles, and transfer learning was performed on different 5 vehicle types.  

#### **Implementation **
1. Install the necessary dependencies from ```requirements.txt``` 
2. Clone  tensorflow models repo from [here](https://github.com/tensorflow/models) (to use LARS optimizer for SimCLR)
3. To run ```pretrain_and_linear_eval.py``` file, several arguments need to be given:

    * ```-a``` : name of the algorithm to be used e.g. simsiam
    * ```-i``` : images directory 
    * ```-n``` : multiplier for image generator (if 2 is given, then an augmented version of data will be added to the whole dataset to becoume double)
    * ```-s``` : unified image size for pretraining 
    * ```-t``` : number of trials (for cross validation)
4. Run ```xml2txt.py``` file to prepare data for transfer learning, the following arguments should be given:
    * ```-i``` : input images directory for the dataset that will be used in transfer learning
    * ```-l``` : xml annotations directory
    * ```-d``` : detination where all images and annotations will be sent (this directory will be used in ```detectionTL.ipynb```)
5. For transfer learning phase, you will give the directory generated from ```xml2txt.py``` in ```detectionTL.ipynb``` notebook.

The initial structire should be something like this:
```
.
├── SSL-project/
│   ├── models/
│   ├── pretraining_dataset/
│   │   └── train/
│   │       ├── class1
│   │       ├── class2
│   │       └── etc
│   ├── object_detection_images/
│   ├── object_detection_annotations/
│   ├── dest_folder/
│   └── ssl_models/
├── data_util.py
├── pretrain_and_linear_eval.py
├── xml2txt.py
├── classes.txt
└── detectionTL.ipynb
```
Where ```models``` is the cloned folder from step 2, ```dest_folder``` will contain the combined and processed images and annotations for transfer learning  (when executing step 4, put this folder path for argument ```-d``` ). The pretraining and linear evaluation of each trial will be saved in a unique experiment name under ```ssl_models```. Each trial will save pretraining stage with its history and a sub folder named ```cls_eval``` will contain all histories of self-supervised and supervised along with their best-scoring model.

