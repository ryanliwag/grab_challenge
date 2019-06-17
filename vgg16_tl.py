from __future__ import print_function, division
import argparse
import os
import time
import utils
import torch
import horovod
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image
import random
import copy


from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help="data mounting point")
parser.add_argument('--training-folder', type=str, dest='train_folder', help="training folder path")
parser.add_argument('--annotation-file', type=str, dest='annotation', help="annotation file path")
parser.add_argument('--meta-file', type=str, dest="meta", help="meta data file path")
parser.add_argument('--training-split', type=float, nargs='?', dest="training_split", const=0.8, help="training and validation split. Default at .8")
parser.add_argument('--learning-rate', type=float, dest="lr", help="set learning rate")
parser.add_argument('--class-train', type=str, dest="class_", help="specify which class to train on (class_ID, type_ID, year_ID, maker_ID)")
parser.add_argument('--nb_epochs', type=int, nargs='?', dest="epochs", const=30, help="Number of epochs to train the model")
parser.add_argument('--batch_size', type=int, nargs='?', dest="batch_size", const=15, help="batch size")
parser.add_argument('--nb_classes', type=int, dest='nb_classes', help="Number of classes")
parser.add_argument('--no_cuda', action="store_false", dest='cuda_switch', help="Dont use CUDA")
args = parser.parse_args()

if args.cuda_switch:
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)

class car_dataset(Dataset):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """
    def __init__(self, files, root_dir, meta_data, image_transform=None):
        
        self.root_dir = root_dir
        self.image_transform = image_transform
        
        #image file names
        self.image_files = [file[-1][0] for file in files]
        
        #Class ID
        self.id = [file[-2][0] - 1 for file in files]
        
        #Class Name
        self.class_name = [meta_data[file[-2][0] - 1][0] for file in files]
        
        #Get Car Year
        self.carYear, self.carYear_ID = utils.get_Year(self.class_name)
        
        #Get Car Maker
        self.carMaker, self.carMaker_ID = utils.get_Maker(self.class_name)
        
        #Get Car Type
        self.carType, self.carType_ID = utils.get_Type(self.class_name)
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path)

        # create numpy array from PIL image
        nparray = np.array(img)
        # create PIL image from numpy array
        img = Image.fromarray(nparray)
        if self.image_transform:
            img = self.image_transform(img)
            
        target = torch.from_numpy(np.array(self.id[idx]))[0]

        sample = {'Image':img, 'class_ID':target, "class_name":self.class_name[idx],
                 'year_ID':self.carYear_ID[idx], 'maker_ID':self.carMaker_ID[idx],
                 'type_ID':self.carType_ID[idx]}
        
        return sample

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, class_type):
                      
        self.indices = list(range(len(dataset)))
        
        self.num_samples = len(self.indices) 
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, class_type)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx, class_type)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx, class_type):
        return dataset[idx][class_type].item()
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
                self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

image_transformers = {'train': transforms.Compose([transforms.Resize((244,244)),
                                                   transforms.RandomRotation(degrees=60),
                                                   transforms.RandomHorizontalFlip(0.8),
                                                   transforms.ColorJitter(brightness=0.8, contrast=0.8),
                                                   transforms.ToTensor()]),
                      'validation': transforms.Compose([transforms.Resize((244,244)),
                                                       transforms.ToTensor()
                                                       ])
                     }
                     
# get hold of the current run
run = Run.get_context()
 
meta_path = os.path.join(args.data_folder, args.meta)
annotations_path = os.path.join(args.data_folder, args.annotation)
training_folder = os.path.join(args.data_folder, args.train_folder)
#Load Meta Data
meta_data = loadmat(meta_path)
meta_data = np.concatenate(meta_data["class_names"][0])

dataset = utils.Load_Images(root_dir = training_folder, 
                            annotations_path=annotations_path, seed=seed, test_split=args.training_split)

training_data = car_dataset(dataset["training"],
                            root_dir = training_folder,
                            meta_data = meta_data,
                            image_transform = image_transformers["train"]
                           )

train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, 
                                           sampler=ImbalancedDatasetSampler(training_data, args.class_))

validation_data = car_dataset(dataset["validation"], 
                             root_dir = training_folder,
                             meta_data = meta_data,
                             image_transform  = image_transformers["validation"])

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

print("Data is Loaded")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU using: {}".format(torch.cuda.get_device_name(0)))

vgg_based = torchvision.models.vgg19(pretrained=True) 

for idx,param in enumerate(vgg_based.parameters()):
    if idx <= 32:
        param.requires_grad = False
    
# Modify the last layer
number_features = vgg_based.classifier[6].in_features
features = list(vgg_based.classifier.children())[:-1] # Remove last layer
features.extend([torch.nn.Linear(number_features, args.nb_classes)])
vgg_based.classifier = torch.nn.Sequential(*features)

vgg_based = vgg_based.to(device)

print(vgg_based)

criterion = torch.nn.CrossEntropyLoss()

optimizer_ft = torch.optim.Adam(vgg_based.parameters(), lr= args.lr, weight_decay=1e-7)

def train_model(model, criterion, optimizer, num_epochs=1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 30)

        training_loss = 0
        validation_loss = 0
        
        training_accuracy = 0
        validation_accuracy = 0

        # Iterate over data.
        for batch_idx, data in enumerate(train_loader):
            inputs = data["Image"]
            labels = data[args.class_]
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.long()
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs  = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # Compute the total loss for the batch and add it to valid_loss
            training_loss += loss.item() * inputs.size(0)
            
            #train accuracy
            (max_vals, arg_maxs) = torch.max(outputs, dim=1) 
            correct_counts = arg_maxs.eq(labels.data.view_as(arg_maxs))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            
            # Compute total accuracy in the whole batch and add to valid_acc
            training_accuracy += acc.item() * inputs.size(0)

            
        #get accuracy
        with torch.no_grad():
            
            model.eval()
            
            for batch_idx, data in enumerate(validation_loader):
                inputs = data["Image"]
                labels = data[args.class_]
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.long()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                validation_loss += loss.item() * inputs.size(0)

                #train accuracy
                (max_vals, arg_maxs) = torch.max(outputs, dim=1) 
                correct_counts = arg_maxs.eq(labels.data.view_as(arg_maxs))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                validation_accuracy += acc.item() * inputs.size(0)

        # Find average training loss and training accuracy
        avg_train_loss = training_loss/len(training_data)
        avg_train_acc = training_accuracy/float(len(training_data))

        # Find average training loss and training accuracy
        avg_valid_loss = validation_loss/len(validation_data)
        avg_valid_acc = validation_accuracy/float(len(validation_data))
        history.append([avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc])

        #logs
        run.log('training_loss', np.float(avg_train_loss))
        run.log('training_acc', np.float(avg_train_acc))
        run.log('validation_loss', np.float(avg_valid_loss))
        run.log('validation_acc', np.float(avg_valid_acc))

        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(epoch + 1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100))
        
        # deep copy the model
        if avg_valid_loss < best_loss:
            print('saving with loss of {}'.format(avg_valid_loss), 'improved over previous {}'.format(best_loss))
            best_loss = avg_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

model_, history = train_model(vgg_based, criterion, optimizer_ft, num_epochs=args.epochs)

os.makedirs('outputs', exist_ok=True)
torch.save(model_.state_dict(), 'outputs/vgg16_{}_weights.pth'.format(args.class_))
torch.save(model_.state_dict(), 'car_data/vgg16_{}_weights.pth'.format(args.class_))