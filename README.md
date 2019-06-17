# Grab AI challenge Submission

Sadly due to time constraints and work in general. I wasn't able to fully finish this project (In all honesty I think this will take ma a couple of weeks at least) but nonetheless I learned alot and had a ton of fun with the challenge.

### Dataset

Ideally at the start I was simply thinking about implementing an ensemble. But after seeing the labeled dataset, I wanted to try a multi task learning approach. 

![summary](https://raw.githubusercontent.com/ryanliwag/grab_challenge/master/Images/data_summary.PNG)

The Problem statement requires the classification algorithm to identify the Car Maker and Type. The dataset is very sparse, there is a low amount of samples per class compared to the feature dimensions. Also resizing this data to smaller images causes us to lose features. To hopefully work around this problem I am trying to implement multi-task learning models. There is 196 labeled classes in the dataset, most of the classes have the type, maker and year included in the label. These labels can hopefully serve as Auxiliary tasks to help train the model.

### Multi-Task Learning

So the motivation behind using this approach was reading through sebastian ruders [article](http://ruder.io/multi-task/]) (must read). The objective is to improve the model's performance on the current task by having it learn of different but related concepts to the original task.

I will be trying 2 concepts I have picked up from the article which is **Hard Parameter Sharing** and  **Model Training with Auxiliary Tasks**. 

### MTL (Hard parameter sharing)

![hard parameter sharing](http://ruder.io/content/images/2017/05/mtl_images-001-2.png)

This model I essentially took a pretrained frozen model (resnet50) and connect it to two task specific layers. These tasks are identifying car Type and Car Maker.

Here is the  [notebook](https://github.com/ryanliwag/grab_challenge/blob/master/Multi-Task%20Learning%20Model.ipynb). Code block at the end to run on your own images

Todo: Try larger image resolutions, create better labels

### MTL (Auxiliary Task Learning) (No more time)