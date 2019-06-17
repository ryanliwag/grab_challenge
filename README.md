# Grab AI challenge Submission

Sadly due to time constraints and work. I wasn't able to finish the project on time but I learned alot and had a ton of fun with the challenge.

### Dataset

Ideally at the start I was simply thing about implementing a ensemble.  But after seeing the labeled dataset, I wanted to try something new. 

  **image insert**

The Problem statement requires the classification algorithm to identify the Car Maker and Type. There is 196 labeled classes in the dataset, most of the classes have a the type, maker and year included in the label.  

### Multi-Task Learning

So the motivation behind using this approach was reading through sebastian ruders article[http://ruder.io/multi-task/] (must read). The objective is to improve the model's performance on the current task by having it learn of different but related concepts to the original task. Also training on single task brings problems of its own such over fitting, imbalanced dataset and too much reliance on the model to learn the features we need. 

I will be trying 2 approaches I have picked up from the article which is **Hard Parameter Sharing** and  **Model Training with Auxiliary Tasks**. 

### MTL (Hard parameter sharing)
[[1]](http://ruder.io/multi-task/index.html#fn1)  



### MTL (Auxiliary Task Learning)


[Handlebars templates](http://handlebarsjs.com/),