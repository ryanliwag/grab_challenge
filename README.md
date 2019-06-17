# My Submission for the Grab AI challenge.

### Problem Statment
Given a dataset of distinct car images, can you automatically recognize the car model and make?
Yea, will give it a try

Here used pytorch running on python 3.6.5 which is the nightly issue.

### Dataset

Dataset had 196 classes. 

### First Model (Transfer Learning VGG16)

So for my first attempt at modeling, I went ahead with using pretrained models.
Configured a weighted random sampler to sample from my imbalanced dataset () and had included pytorch prebuild image transformers hopefully to prevent some overfitting.



1. First ordered list item
2. Another item

### Second Model (Multi Task Learning Model)



#learning through hints https://www.sciencedirect.com/science/article/pii/0885064X9090006Y?via%3Dihub


image: https://www.google.com/search?q=multi+task+learning&source=lnms&tbm=isch&sa=X&ved=0ahUKEwic8OiK4u7iAhVHUd4KHQ6JA2sQ_AUIECgB&biw=1895&bih=952#imgrc=HWg77j4p2pjDNM:
[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].
![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)


# How to test on your own Dataset

#ToDo First Model:
    - Read Import Data (Done)
    - Visual Analysis (Done)
    - Baseline on Simple and Complex models

#ToDo Second Model (MTL):

08/06/2018 (Baseline on Simple and Complex Models)
08/06/2018 (Try Training Simple Hard parameter sharing MTL)


#Running in Microsoft azure ML:
1.Create workspace (https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-run-cloud-notebook)

2.


# 1st Approach (Transfer Learning with Complex and Simple Models)

# 2nd Approach (Multi Task Learning)

# 3rd Approach (Solve)