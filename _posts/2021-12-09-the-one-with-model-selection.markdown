---
layout: post
title:  // The one with Model Selection (Part 2)
date:   2021-12-09
categories: post
---

---
**Note**: *This post is a script of my understanding of an article written by one of my favorite researcher, **Dr. Sebastian Raschka**, Assistant Professor of Statistics at the University of Wisconsin-Madison, US. [Link to his paper.](https://arxiv.org/abs/1811.12808)*

---



<br>
## <span style="color:red">// Recap: Need of Hyperparameters

  - As stated by Dr. Raschka in his paper (link above), the **hyperparameters** are variables that are used by the learning algorithm in order to compute the best **feature  parameters**. They are needed to be specified apriori and externally by the experimenter.
  
  - Hyperparameters are not governed by any fixed protocol. The experimenter needs empirical evaluation to gather appropriate value(s) for them.
  
  - **As and when we change a hyperparameter value, a new model gets manifested. Out of all these *n* models, the process of selecting the best one is called Model Selection.**
  
  - Model selection is nothing but hyperparameter optimization. Since, we are selecting appropriate hyperparameters, we are exploring their values that would help our machine learning algorithm to find the best approximation of parameters to underlying population's feature set.

<br>

## <span style="color:blue;">// Model Selection techniques:

# 1. 3-way Holdout method:
Recall, in **Holdout method**, we used to partition our entire dataset into two parts -- train split and test split. In 3-way Holdout, we will extend the idea a bit further:
	
  - **<u>How to use it</u>**: 
    - We will experiment by further dividing the train set into two sub-parts: **a train sub-split and a validation sub-split.** 
	- While train sub-split would be used for model training, the validation sub-split would be used for selection of the best model among *n* models (mentioned in the *Recap: Need of Hyperparameters*).
	- But how is this validation sub-split help select the best model? As we have already discussed, apart from the feature parameters, there are certain hyperparameters that need fine-tuning in order to get as  accurate feature parameters as possible. This validation sub-split would help us select the best hyperparameter values  by running the machine learning algorithm over and over again using different values of hyperparameters.  Since each modification in the value of hyperparameters result a new model, we would finally select that model which would result in the best feature parameters among all models.
	- It is obvious that there would be certain metric(s) in order to measure the performance of each model. These could be accuracy, loss, F1-score etc.
	- **Thus, we would compare each model's performace, and select the model that is best than its peers.**
	- Finally, after hyperparameter selection and best model selection, we use the test split for a final evaluation of the model on unseen data. Optionally, we before testing, we could merge both train sub-split and validation sub-split. <br><br>
  
  - **<u>What may go wrong</u>**: While 3-way Holdout method sounds nice, it has a brief pitfall too, highly related to those of Holdout method of model evaluation. For example, if the splitting is not done in stratified manner, then some bias-variance may get introduced in the resultant model. Secondly, if dataset in hand is less, then the experimenter should consider augmentation of data to use 3-way Holdout method. That said, one could easily use 3-way Holdout method for model selection if data deficiency is not an issue.
  
  
	<center>
	<figure>
	<img src="/research/images/images/the-one-with-model-selection/3-way-holdout.SVG" />
	<figcaption>Fig.: 3-way Holdout method conceptual art.</figcaption>
	</figure>
	</center>
	<br>
  
# 2. k-Fold Cross Validation (k-Fold CV):
k-Fold Cross Validation could be considered as the most promising method for model selection, particularly when there is deficiency of data.

  - **<u>How to use it</u>**: 
    - The key idea is to **give each data example a chance to get tested on.**
	- First, divide the dataset into k parts (say k=5). Then, execute the following algorithm for k iterations (or so we call ***fold*** here):
	  <ol>
	    <li> Select <b>k-1</b> parts as the train split, and the left-over part as test split.
		<li> Fit a model using the train split.
		<li> Evalute the fitted model on the left-over test split. Compute its performace using any appropriate metric.
		<li> Store this performance value somewhere safe. We would need it later.
	- After k iterations, the final performance value is computed as the aggregation of all the k fold performace scores.
	
	$$ performance_{\text{k-fold}} = \frac{1}{k}\sum_{i=1}^{k}performance_{i} $$
	
	<center>
	<figure>
	<img src="/research/images/images/the-one-with-model-selection/k-fold.SVG" />
	<figcaption>Fig.: k-Fold Cross Validation conceptual art.</figcaption>
	</figure>
	</center>
	<br>

  - **<u>Two special cases of k-Fold CV</u>**:
    - The k-Fold CV method seems to be quite similar to Repeated Holdout method, however, there is a vital difference. While in case of Repeated Holdout, the numerous test splits may have a chance of overlapping, i.e., instances in one test split may occur in other splits too. On the contrary, there is no chance of this to happen in k-Fold Cross Validation (see figure). This brings us to study two special cases of k-Fold:
	  - **_2-Fold Cross Validation_**: A variant of k-Fold CV, where **k=2**. Thus, there would be two iterations. The dataset would be divided in half. For iteration 1, the first half could be considered as train split and the second half as test split. For iteration 2, the second half could be considered as train split (if the second half is considered as test split in iteration 1), and the first as test split. 
	  
	    
		<center>
		<figure>
		<img src="/research/images/images/the-one-with-model-selection/2-fold-cv.SVG" />
		<figcaption>Fig.: 2-Fold Cross Validation concept.</figcaption>
		</figure>
		</center>
		<br>
		
	  - **_Leave-one-out-cross-validation (LOOCV)_**: In this variant, we set **k=no. of instances in the dataset (say *n*)**. This gives each individual instance to become a test split. Though it may sound good, but it is quite expensive computationally (if the dataset is huge in size). Suppose, we have *n=1000*, then LOOCV would execute for 1000 iterations. One more issue often arises -- **high variance, low bias.** Since the train split is in abundance, thus, it takes care of the bias. But, since we have only ONE instance as test split in each iteration, it leads LOOCV to give varied results.
	  
	  
		<center>
		<figure>
		<img src="/research/images/images/the-one-with-model-selection/loocv.SVG" />
		<figcaption>Fig.: Leave-one-out-cross-validation concept.</figcaption>
		</figure>
		</center>
		<br>
	
  
		```python
		## LOOCV on Digits data
		## computationally very expensive on large dataset

		import numpy  as np
		import matplotlib.pyplot as plt
		from sklearn import datasets
		import warnings
		import time
		warnings.filterwarnings('ignore')
		plt.style.use('ggplot')
		plt.rcParams['figure.figsize']=[8,5]
		from sklearn.model_selection import LeaveOneOut
		from sklearn.linear_model import LogisticRegression
		##------------------------------------------
		digits = datasets.load_digits()
		digits_data = digits.data
		digits_label = digits.target
		##------------------------------------------
		start = time.time()
		loo = LeaveOneOut()
		score = []
		for train, test in loo.split(digits_data):
			digit_train, digit_test = digits_data[train], digits_data[test]
			digit_train_y, digit_test_y = digits_label[train], digits_label[test]
			clf = LogisticRegression(solver='liblinear')
			clf.fit(digit_train, digit_train_y)
			score.append(clf.score(digit_test, digit_test_y))

		end = time.time()
		print("Time taken = ", end - start, "seconds")
		print("Mean Accuracy\t: ", np.mean(score))
		
		plt.bar(np.arange(len(score)), score, align='center')
		plt.xlabel("iterations")
		plt.ylabel("accuracy score")
		plt.title("K-Fold CV with k = #dataset")
		plt.show()
		```
		Time taken =  276.3200423717499 seconds\\
		Mean Accuracy	:  0.9643850862548692
		<center>
		<figure>
		<img src="/research/images/images/the-one-with-model-selection/loocv_digits.svg" />
		<figcaption>Fig.: Leave-one-out-cross-validation graph on DIGITS dataset.</figcaption>
		</figure>
		</center>
		<br>

		```python
		## K-Fold on Digits data with k=10 
		
		import numpy  as np
		import matplotlib.pyplot as plt
		from sklearn import datasets
		import warnings
		import time
		warnings.filterwarnings('ignore')
		plt.style.use('ggplot')
		plt.rcParams['figure.figsize']=[8,5]
		from sklearn.model_selection import StratifiedKFold
		from sklearn.linear_model import LogisticRegression
		##------------------------------------------
		digits = datasets.load_digits()
		digits_data = digits.data
		digits_label = digits.target
		##------------------------------------------
		start = time.time()
		kfold = StratifiedKFold(n_splits=10)
		score = []
		for train, test in kfold.split(digits_data, digits_label):
			digit_train, digit_test = digits_data[train], digits_data[test]
			digit_train_y, digit_test_y = digits_label[train], digits_label[test]
			clf = LogisticRegression(solver='liblinear')
			clf.fit(digit_train, digit_train_y)
			score.append(clf.score(digit_test, digit_test_y))

		end = time.time()
		print("Time taken = ", end - start, "seconds")
		print("Mean Accuracy\t: ", np.mean(score))
		
		plt.bar(np.arange(len(score)), score, align='center')
		plt.xlabel("iterations")
		plt.ylabel("accuracy score")
		plt.title("K-Fold CV with k = 10")
		plt.show()
		```
		Time taken =  1.5524160861968994 seconds\\
		Mean Accuracy	:  0.9259745499689634
		<center>
		<figure>
		<img src="/research/images/images/the-one-with-model-selection/10fold_digits.svg" />
		<figcaption>Fig.: k-Fold CV graph on DIGITS dataset. Notice the time taken by LOOCV and k-Fold with n=10.</figcaption>
		</figure>
		</center>
		<br>

		```python
		## K-Fold on Digits data with k=2

		import numpy  as np
		import matplotlib.pyplot as plt
		from sklearn import datasets
		import warnings
		import time
		warnings.filterwarnings('ignore')
		plt.style.use('ggplot')
		plt.rcParams['figure.figsize']=[8,5]
		from sklearn.model_selection import StratifiedKFold
		from sklearn.linear_model import LogisticRegression
		##------------------------------------------
		digits = datasets.load_digits()
		digits_data = digits.data
		digits_label = digits.target
		##------------------------------------------
		start = time.time()
		kfold = StratifiedKFold(n_splits=2)
		score = []
		for train, test in kfold.split(digits_data, digits_label):
			digit_train, digit_test = digits_data[train], digits_data[test]
			digit_train_y, digit_test_y = digits_label[train], digits_label[test]
			clf = LogisticRegression(solver='liblinear')
			clf.fit(digit_train, digit_train_y)
			score.append(clf.score(digit_test, digit_test_y))

		end = time.time()
		print("Time taken = ", end - start, "seconds")
		print("Mean Accuracy\t: ", np.mean(score))
		
		plt.bar(np.arange(len(score)), score, align='center')
		plt.xlabel("iterations")
		plt.ylabel("accuracy score")
		plt.title("K-Fold CV with k = 2")
		plt.show()
		```
		Time taken =  0.11986732482910156 seconds\\
		Mean Accuracy	:  0.9031805941271048
		<center>
		<figure>
		<img src="/research/images/images/the-one-with-model-selection/2fold_digits.svg" />
		<figcaption>Fig.: k-Fold CV graph on DIGITS dataset. Notice the time taken by LOOCV, k-Fold with n=10 and k-Fold with n=2.</figcaption>
		</figure>
		</center>
		<br>
		
		
  - **<u>Model Selection via k-Fold</u>**
    <ol>
	  <li> Similar to 3-way Holdout method, first split the entire dataset into train split and test split.
	  <li> Take the train split in hand, set a value of <b>k</b>. Lets say k=10 ("a sweet spot").
	  <li> Partition the entire train split into k parts.
	  <li> For iteration <i>i</i>, select <i>i</i>th part as validation sub-split and rest as train sub-split. Train the model using the train sub-split and validate it (and the hyperparameter values) on the validation sub-split. 
	  <li> After k iterations, average out the performance score by taking the mean of performance scores obtained at the end of each iteration. Optionally, merge entire train sub-parts and validation sub-split and retrain the selected model on the merged train split.
	  <li> Finally, evaluate the model's performance on the unseen test split that was done in Step 1.
	
	<br><br>
	  
  - **<u>Verdict on k-Fold Cross Validation</u>**:
    - Sometimes, the Holdout methods are considered over k-Fold, even though k-Fold provides a better approach to model evaluaiton and selection. Then why so?
	- Although we are still hungry for data, in the current world, to train an industry-level model, the companies already collect ample amount of data for model training. In this case, it is assumed that a Holdout approach would probably give a sufficiently accurate result, with little time consumption. On the other hand, whilst k-Fold Cross Validation COULD give a minor improved result, it would consume a lot of time, which is of the essence. With huge dataset in hand, even current models sometime may take weeks to get trained! With this situation, we cannot afford to iterate model training process for 'k' number of times.
	- Also, it is evident that huge datasets theoretically benefit the performance of models. Suppose we have 1 million instances in a dataset. In this case, if we split the dataset into 70/30 ratio or 80/20 ratio, it would not harm the performance drastically, since we will have enough training instances to fit the model up to its capacity, along with enough testing instances to evaluate our model’s performance appropriately. So, we will actually take care bias and variance issue with large datasets and Holdout method approach.
	
	<br>

## <span style="color:brown;">// Key take-aways:
  - **Model Selection** is a crucial step while building a model. If we do not select appropriate model hyperparameters, we would. in-turn, select feature parameters that are not suited for our underlying population, and thus, we will have a **_not-so-good_** model in the end, consuming a lot of time and resources, and providing average result (or sometimes, less).
  
  - We studied different ***model evaluation methods***, starting from naive Resubstitution Method, which is not recommended due to its overfitting nature. Then we studied about different aspects of Holdout Method (most recommended for huge dataset). After that, we looked at the intuition behind the Bootstrap Method. A great idea to build up deficit dataset using sampling with replacement, but should be used with small datasets only due to its computational inefficiency.
  
  - Further, we studied ***model selection methods***, where we looked at 3-way Holdout Method, an extension of simple Holdout method, for model selection, where we sub-divide the train split into train sub-split and validation sub-split. The train sub-split is used for model training and fitting, while the validation sub-split helps select the best model hyperparameter values. Finally, we looked at the most promising method, the k-Fold Cross Validation, hugely used with small datasets. But we also considered its shortcoming — computationally very expensive if used over large datasets.
  
<br>
## <span style="color:purple;">// References:
 [S Raschka et al., *Model evaluation, model selection, and algorithm selection in machine learning*, arXiv preprint arXiv:1811.12808, 2018 - arxiv.org](https://arxiv.org/abs/1811.12808)

<br>

---
---

[//]: "------------------------------ THE ONE WITH MODEL SELECTION --------------------------------"