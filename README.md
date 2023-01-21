# Novel-Deep-Learning-IDS-Code
This is the code repository for my research project: "Creating an Effective Intrusion Detection System Using a Novel Deep Learning Algorithm". Specifically, the code concerns the method section of my research.

# Datasets
The datasets used for this research project are [Kyoto 2006+](http://www.takakura.com/Kyoto_data/) and [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) datasets. Binary encoding was performed on the categorical features.

<i>Kyoto 2006+:</i>

5 million datapoints from the 25 million datapoints of the Kyoto 2006+ 2007 subset were sampled to reduce computational resources required. Kyoto 2006+ contains 14 numerical features, 9 categorical features, and 1 target variable. 

<i>UNSW-NB15:</i>

The 4 different CSV files of UNSW-NB15 were combined to create a large UNSW-NB15 dataset with 930,000 data points. The 41 numerical features and 6 categorical features of UNSW-NB15 were utilized, with the ‘attack_cat’ feature being excluded since it corresponded with the target variable.

# [Machine Learning Models](https://github.com/BOLTZZ/Novel-Deep-Learning-IDS-Code/tree/main/Machine%20Learning%20Models)
Machine learning models were created to achieve basline accuracies to compare against the deep learning models and the novel deep learning algorithm. Z-score normalization was perfomed before implementing the machine learning models. These models were trained on 3 different types of data categories: only numerical features, only categorical features, or categorical and numerical features. Confusion matricies were printed out for each model to get a sense of the model's performance.

<i>Logistic Regression:</i>

One of the simplest machine learning models, logistic regression, was applied to the datasets. As expected, logistic regression had the lowest accuracies out of all the models. Different solvers were utilized to maximize accuracy.

<i>Decision Tree:</i>

Decision trees performed better than the logistic regression models. 

<i>Random Forest:</i>

Random forest models were trained with 70 trees. This showed the difference between 1 tree (decision tree) and 70 trees (random forest). Random forests performed the best out of all the machine learning models.

# [Deep Learning Models](https://github.com/BOLTZZ/Novel-Deep-Learning-IDS-Code/blob/main/Deep%20Learning%20Models/Final%20Deep%20Learning%20Models.ipynb)
3 different deep learning models (MLP, VAE, CNN) were utilized. Unique architecture was created for the MLP and CNN.

<i>Multi-layer Perceptron:</i>

The MLP is a feed-forward neural network. The MLP was trained for 10 epochs with a batch size of 64. Accuracy and loss were obtained to evaluate the performance.

<i>Variational Autoencoder:</i>

The VAE is a type of generative model, generating new data from the training data. The VAE is trained to learn a low-dimensional representation of the input data. The VAE was trained for 50 epochs with a batch size of 2,250. Accuracy and loss were obtained to evaluate the performance.

<i>Convolutional Neural Network:</i>

The CNN is a type of neural network effective at processing grid-like data, like images. The CNN was trained for 10 epochs with a batch size of 64. Accuracy and loss were obtained to evaluate the performance.

# [Feature Selection/Feature Comparision](https://github.com/BOLTZZ/Novel-Deep-Learning-IDS-Code/blob/main/Feature%20Selection%20Process.ipynb)
To get a sense of the most impactful features per dataset, random forest classifcation was performed. Before, the classification, the categorical features were encoded with label encoding to preserve the original columns. In addittion, features were compared between datasets to search for common features.

# Novel Deep Learning Algortihm

A deep learning algorithm with novel architecture was created based of the CNN model since it performed the best. This algorithm only utilized the 9 features selected from the feature selection/feature comparision. Label encoding was performed on the categorical features to preserve the columns for the architecture of the model. This model was trained for 10 epochs with a batch size of 64. The accuracy and loss were obtained to evaluate the model.

# Visualizing the Workflow
![Visual of the workflow described.](https://github.com/BOLTZZ/Novel-Deep-Learning-IDS-Code/blob/main/Images/workflow.png)
