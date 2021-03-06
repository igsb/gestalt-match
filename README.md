# Gestalt Match
The goal of Gestalt Match is matching two patients with the same **unknown** rare monogenic disorders. We utilize facial recognition approach such as FaceNet or ArcFace to extract the facial embeddings from patient's facial photo and further perform clustering analysis.

## Requirement
  * python version >= 3.5
  * FaceNet
  * Face2Gene account
  
### Configuration
Fill up the **config.ini** to access Face2Gene website to fetch the photos and meta data in JSON format. You can check the example in `config.ini.SAMPLE`. The format is as follow:
```
[Download]
user = your face2gene account
password = your face2gene pwd
# the folder to save all the photo and meta data
data_path = data
# the folder to save the data in data folder you specify above
project_name = pedia
# input_path: JSON files of PEDIA cohort, please use LAB api
# to download or contact Tzung-Chien Hsieh if you don't have
# credential
input_path = pedia
```

### Reminder (important)
Please do not add the following files into repository
* photo (.jpg, .png, .svg)
* .json files which contain the patient data
* .npy file generated by FaceNet or the other facial recognition approach
* config.ini

Please use private folder in sciebo or the server in IGSB to store the patient data. Specify the link or path in config file to make sure only the developers who has the credential are able to access and download the patient data.

## Todo
* setup FaceNet
* setup ArcFace
* setup side files such as the pretrained model
