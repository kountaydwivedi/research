---
layout: post
title:  // The one with Model Evaluation (Part 1)
date:   2021-10-31
categories: post
---

---
**Note**: *This post is a script of my understanding of an article written by one of my favorite researcher, **Dr. Sebastian Raschka**, Assistant Professor of Statistics at the University of Wisconsin-Madison, US. [Link to his paper.](https://arxiv.org/abs/1811.12808)*

---



<br>
## Abstract:
The ultimate target of any data scientist is to deduce a model that is almost a *perfect*  fit to the underlying original distribution of the population. While, developing a model that could imitate the original distribution flawlessly is almost infeasible (since we always do not have entire population to train the  model), a model that *almost* mimics the underlying distribution is quite feasible. 

Getting a high accuracy should not be the only purpose of any data scientist. If a biased dataset or small dataset is used for training a model, that model might perform awesome on the in-hand dataset, but it might fail to produce accurate result in the real world with generalised instances. Hence, machine learning practitioners/data  scientists are always tuning the model's *"parameters and hyperparameter"*. since parameter assignement is unfortunately not in our hand (unless we manually assign weights to every feature in the dataset), the only thing that we have in our hand that we could optimise according to our distribution is the set of hyperparameters in our dataset.

> Just fitting a model haphazardly, to get a high accuracy, does not make it a **good predictor**. Hyperparameter tuning is a must-to-do step, since accuracy is not ALL what we want. We need a model that is awesome in **generalization**. No one likes a rote-learner !!

<br>

## Model evaluation techniques:

# 1. Resubstitution evaluation:
Also known as Residual Method, it is the most basic method of model evaluation, and thus highly non-recommended.
	
  - **<u>How to use it</u>**: The idea is to use all the available data in hand to fit the model using any method deemed appropriate by the scientist. Thereafter, evaluate the model using the same available data (over which the model is trained) on the basis of the ground truth.
  
  - **<u>Why it is not recommended</u>**: The answer is quite simple. We are evaluating  the model on the same data on which it has been trained, leaving ***no unseen data*** over which we could perform the  evaluation. The model will obviously return high accuracy over the seen data. However, this accuracy should actually be considered as the training accuracy (a.k.a resubstitution accuracy) rather than evaluation accuracy. Thus, if Resubstitution method is followed, one should expect a high accuracy in training, but ***poor generalization or prediction accuracy*** when real-time or unseen data is fed. This is also a common reason that a model shows high variance over unseen data.
  
  
	> A student should not be asked the same question in the exam that he had been taught in the class.
  
  
	  ```python
	  '''
	  Implementing Resubstitution method in python
	  Dataset used: IRIS dataset
	  '''
	  #--------------------------------------------------------
	  import numpy as np
	  from sklearn import datasets
	  from sklearn.linear_model import LogisticRegression
	  #--------------------------------------------------------
	  iris = datasets.load_iris()
	  X = iris.data
	  y = iris.target
	  #--------------------------------------------------------
	  clf = LogisticRegression(solver='liblinear')
	  clf.fit(X, y)
	  print("Accuracy score = ", clf.score(X, y))
	  ```
  
  <br>
  
# 2. Holdout evaluation:
Holdout evaluation actually tries to cover the disadvantages of Resubstitution Method, but brings on the desk some of its own caveats.
  - **<u>How to use it</u>**: The key idea here is to split the dataset in hand into two parts, ***train split*** and ***test split***. The model is trained on the train split. Later, after thorough training and all parameters and hyperparameters optimisation, the model is finally  evaluated on the unseen the test split.
  
  - **<u>Shortcomings that may befall</u>**:
    - The first shortcoming that may occur in a Holdout evaluation is actually related with the way of "splitting". Generally, there are two ways a dataset could be splitted: ***simple random split*** and ***stratified random split***. If we opt for simple random split in Holdout evaluation, the ratio of each sub-classes present in the dataset may  get distorted, i.e., the ratio of different sub-classes in the whole dataset in hand may not be the same in train split and test split after simple random splitting.
	<br>
	However, if we opt for stratified random split, the ratio of all the sub-classes present in the entire dataset is kept intact in train split  as well as in test split. Now, this is a recommended approach to follow, as stratification helps in eliminating the bias-variance from the train and test split.
	
	- The second shortcoming, unfortunately, is quite hard to resolve. Its called ***deficiency of data***. What if the dataset we have in hand, does not contain enough instances for the model to get trained and/or evaluated? This is an ongoing problem that machine learners and data scientists face, and the only solution to it is ***"the more the merrier"*** (although, lately there are few advanced techniques available such as Zero-shot, One-shot or Few-shot learning, where none to very few intances are enough for training a model, but more on this later).
	
	- The third shortcoming is actually a general caveat, but there is no harm mentioning here. It is called ***dataset imbalance***. A condition, where the ratio of sub-classes within the dataset is significantly different, thus, making the sample distribution biased towards the sub-class which has higher number of instances in the dataset. To avoid this situation, the first solution would be to collect as many samples of all sub-classes as possible to avoid imbalance. Secondly, use data augmentation techniques (GANS, SMOTE etc.) to build synthetic data. 
	<br>
	> A very bright idea, indeed. Should be handled with care.
  
		```python
		'''
		Implementing Holdout Evaluation on IRIS dataset. The method used is train_test_split().
		In this method, the user needs to pass train_size or test_size as kwargs to determine the
		ratio of train and test split. Along with this, the user can pass whether the split he 
		wants is simple or stratified according to the ground truth.
		'''
		#-------------------------------------------------------------------------
		import numpy as np
		from sklearn import datasets
		from sklean.linear_model import LogisticRegression
		from sklearn.model_selection import train_test_split
		#-------------------------------------------------------------------------
		iris = dataset.load_iris()
		X = iris.data
		y = iris.target
		#-------------------------------------------------------------------------
		xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify = y)
		clf = LogisticRegression(solver='liblinear')
		clf.fit(xtrain, ytrain)
		print('Accuracy score', clf.score(xtest, ytest))
	```
	<br>
	
# 3. Repeated Holdout Method (modified Holdout evaluation):
A modified and robust variant of Holdout Evaluation method, it actually performs ***multiple iteration of Holdout method***, each time with a different seed value, thus, shuffling the entire dataset for train and test split in each iteration. After each iteration, the score is computed and stored. In the end, the average of all score is considered.

<center>
<img src="/research/images/equations/the-one-with-model-evaluation/eq1.svg" />
</center>

<br>
```python
'''
Repeated Holdout evaluation method over IRIS dataset.
train-test split is done with 50% data in each split.
'''
#--------------------------------------
import numpy  as np
import matplotlib.pyplot as plt
from sklearn import datasets
#--------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#--------------------------------------
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['figure.figsize']=[8,5]
#--------------------------------------
iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

clf = KNeighborsClassifier(n_neighbors=3)
accuracies = []
for i in range(50):
    xtrain, xtest, ytrain, ytest = train_test_split(iris_data, iris_label, test_size = 0.5, stratify=iris_label)
    score = clf.fit(xtrain, ytrain).score(xtest, ytest)
    accuracies.append(score)
#--------------------------------------
print("Mean accuracy: ", np.mean(accuracies))
plt.bar(np.arange(len(accuracies)), accuracies)
plt.xlabel("iterations")
plt.ylabel("accuracy score")
plt.title("Repeated Holdout using 50/50 train-test split")
plt.show()
```

<center>
<img src="/research/images/equations/the-one-with-model-evaluation/repeated-holdout-1.svg" />
</center>

<br>

```python
'''
Continuing the above code.
Repeated hold out validation over iris  dataset but using 
90/10 train-test split.
'''
#--------------------------------------
clf = KNeighborsClassifier(n_neighbors=3)
accuracies = []
for i in range(50):
    xtrain, xtest, ytrain, ytest = train_test_split(iris_data, iris_label, test_size = 0.1, stratify=iris_label)
    score = clf.fit(xtrain, ytrain).score(xtest, ytest)
    accuracies.append(score)
#--------------------------------------
print("Mean accuracy: ", np.mean(accuracies))
plt.bar(np.arange(len(accuracies)), accuracies)
plt.xlabel("iterations")
plt.ylabel("accuracy score")
plt.title("Repeated Holdout using 90/10 train-test split")
plt.savefig("repeated-holdout-2.svg", format="svg")
plt.show()
```
<center>
<img src="/research/images/equations/the-one-with-model-evaluation/repeated-holdout-2.svg" />
</center>
<br>

Notice the accuracy bars. When ***train-test-split is 50/50***, they never reach up to their full capacity (due to lack of train set), but have very low variability throughout. Contrary, when ***train-test-split is 90/10***, we get awesome train accuracy, but the evaluation accuracy decreases in many iterations.

<br>
# 4. Bootstrap evaluation
We discussed about the deficiency in dataset problem in Holdout method. The Repeated Holdout evaluation somewhat overcomes this shortcoming, but not entirely.
<br>
The bootstrap method is an alternative approach to model evaluation, which is often considered when there is a deficiency in the dataset. It was proposed to take care of the problem of small datasets.
  - **<u>How to use it</u>**: The concept of bootstrapping is to utilise the available dataset instances again and again. That is: ***repeatedly sample from the original dataset with replacement***.
  <br><br>
  Lets say we have *n* instances in our dataset, or the *bag of instances*. We have to create a train-test split with *p* and *q* instances in them respectively. We will start with train split. We will pick an instance from the *bag of instances*, make a copy of it in our train split bag, and put it back in the *bag of instances*. Next, we again pick an instance from the *bag of instances*, disregarding the fact that an instance present there is already copied to out train split bag. We repeat the same process unless we have *p* instances in train split bag. Now, we will collect those instances that did not make to train split bag, and collectively put them in test split bag. We call these instances ***leftover instances***. Note that the number of leftover instances would decide the value of *q*. Also, we can make *p* as large as possible, though a huge number will ultimately dissolve the reason of using bootstrap (since most of the instances would be just the repetition). The train split bag is actually called ***bootstrap sample*** and the test split bag is called ***out-of-bag sample***. Finally, we fit a model on the bootstrap sample, and evaluate  it using out-of-bag sample.
  <br><br>
  If we repeat the above process for *b* iterations, we can take the average of all the scores and consider it as the accuracy of the model, just like we did in Repeated Holdout evaluation method.
  
  <center>
  <img src="/research/images/equations/the-one-with-model-evaluation/eq2.svg" />
  </center>
  
  - **<u>An intuitive drawback</u>**: Bootstrap, although seems great for small datasets, suffers from ***pessimistic bias***. Learn more about it in my **bias-variance tradeoff** post.
  <br>
  
	> Why can not a man lift himself by pulling up on his boot-straps?

<br>

## Key take-aways:
After going through various model evaluation method, the question arises, which method to use? Well:

  - The **Resubstitution evaluation** method is a very naive method and thus, not recommended, due to its ignorance of the importance of evaluation over unseen data, thus bringing in very high variance.
  
  - The **Holdout evaluation** method could be considered as a standard model evaluation method. The drawback comes when there is deficiency of dataset. Thus, it could be concluded that Holdout evaluation (**Repeated Holdout method**, especially) is best suited for model evaluation when dataset is huge in size.
  
  - Lastly, we saw the **Bootstrap evaluation** method, and its workflow. Undoubtedly, it is a nice and intuitive approach to follow when there is a deficiency of dataset. However, it does not state Bootstrap as the best method to use, since it could lead the model to a pessimistic bias.

<br>

## References
 S Raschka et al., *Model evaluation, model selection, and algorithm selection in machine learning*, arXiv preprint arXiv:1811.12808, 2018 - arxiv.org

<br>

---
---

[//]: "------------------------------ THE ONE WITH MODEL EVALUATION --------------------------------"