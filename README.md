# Grab AI challenge Submission

Sadly due to time constraints and work. I wasn't able to finish the project on time but I learned alot and had a ton of fun with the challenge.

### Dataset

Ideally at the start I was simply thinking about implementing an ensemble. But after seeing the labeled dataset, I wanted to try a multi task learning approach. 

![summary](https://raw.githubusercontent.com/ryanliwag/grab_challenge/master/Images/data_summary.PNG)


The Problem statement requires the classification algorithm to identify the Car Maker and Type. There is 196 labeled classes in the dataset, most of the classes have a the type, maker and year included in the label.  These labels can serve as Auxiliary tasks to help train the model.

### Multi-Task Learning

So the motivation behind using this approach was reading through sebastian ruders [article](http://ruder.io/multi-task/]) (must read). The objective is to improve the model's performance on the current task by having it learn of different but related concepts to the original task. Also training on single task brings problems of its own such over fitting on the imbalanced dataset.

I will be trying 2 concepts I have picked up from the article which is **Hard Parameter Sharing** and  **Model Training with Auxiliary Tasks**. 

### MTL (Hard parameter sharing)

![hard parameter sharing](http://ruder.io/content/images/2017/05/mtl_images-001-2.png)


### MTL (Auxiliary Task Learning)