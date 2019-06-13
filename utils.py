from PIL import Image
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
import re
import os

def RGB_check(filepath):
    '''
    Check if Image is RGB
    '''
    im = Image.open(filepath)
    im = np.asarray(im)

    if len(im.shape) == 3:
        return True

    return False

def Load_Images(root_dir, annotations_path, 
                seed, train_split = .8, 
                dataset_shuffle=True):
    '''
    Take annotations, files, and metafdata

    return a dictionary with values.

    split to train and val and test
    '''
    annotations = loadmat(annotations_path)

    file_names = [file for file in annotations["annotations"][0] if RGB_check(os.path.join(root_dir, file[-1][0]))]
    
    if dataset_shuffle:
        np.random.seed(seed)
        np.random.shuffle(file_names)

    nb_samples = len(file_names)
    t_idx = int(nb_samples * train_split)

    train_samples = file_names[:t_idx]
    validation_samples = file_names[t_idx:]

    dataset = {"training":train_samples, "validation":validation_samples}

    return dataset


def Label_encoder(labels):

    fitter = preprocessing.LabelEncoder()
    encoded_values = fitter.fit_transform(labels)

    return encoded_values


def get_Year(car_classes):

    #get Year
    carYear = [car.split()[-1] for car in car_classes]
    carYear_ID = Label_encoder(carYear)

    return carYear, carYear_ID


def get_Type(car_classes):

    car_types = ["Coupe", "Sedan", "Cab",
                 "Beetle", "SUV", "Van",
                 "SuperCab", "Convertible",
                 "Minivan", "Hatchback",
                 "Wagon"]
    
    carType = []
    for cars in car_classes:
        isclass = False
        for type_ in car_types:
            
            #List is arranged so that more generalized types are seen first
            if re.search(r"\b{}\b".format(type_), cars, re.IGNORECASE):
                isclass = True
                carType.append(type_)
                break
        
        if not isclass:
            carType.append("no_type")

    carType_ID = Label_encoder(carType)

    return carType, carType_ID

def get_Maker(car_classes):

    makers = ["AM General Hummer", "Acura", "Aston Martin", "Audi" , "BMW", "Bentley", "Bugatti Veyron 16.4","Buick","Cadillac",
             "Chevrolet", "Chrysler", "Daewoo", "Dodge", "Eagle Talon", "FIAT", "Ferrari", "Ford", "GMC", "Geo Metro Convertible", 
             "HUMMER", "Honda", "Hyundai", "Infiniti", "Isuzu Ascender", "Jaguar", "Jeep", "Lamborghini", "Land Rover", 
             "Lincoln Town Car", "MINI Cooper Roadster", "Maybach Landaulet", "Mazda Tribute", "McLaren MP4-12C", "Mercedes-Benz", 
             "Mitsubishi Lancer","Nissan", "Plymouth Neon", "Porsche Panamera", "Ram C/V", "Rolls-Royce", "Scion", "Spyker", "Suzuki", 
             "Tesla", "Toyota", "Volkswagen", "Volvo", "smart fortwo", "Fisker Karma Sedan"]

    carMaker = []
    for cars in car_classes:
        istype = False
        for maker in makers:
            if re.search(r"\b{}\b".format(maker), cars):
                istype = True
                carMaker.append(maker)

    carMaker_ID = Label_encoder(carMaker)

    return carMaker, carMaker_ID