# Deep Learning IDS Code
This is the code repository for my research project: "Creating an Effective Intrusion Detection System Using a Deep Learning Algorithm". Specifically, the code concerns the method section of my research.

Preprint can be found [here]()

# Datasets
The datasets used for this research project are [Kyoto 2006+](http://www.takakura.com/Kyoto_data/) and [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) datasets. Binary encoding was performed on the categorical features.

<i>Kyoto 2006+:</i>

5 million datapoints from the 25 million datapoints of the Kyoto 2006+ 2007 subset were sampled to reduce computational resources required. Kyoto 2006+ contains 14 numerical features, 9 categorical features, and 1 target variable. 

<i>UNSW-NB15:</i>

The 4 different CSV files of UNSW-NB15 were combined to create a large UNSW-NB15 dataset with 930,000 data points. The 41 numerical features and 6 categorical features of UNSW-NB15 were utilized, with the ‘attack_cat’ feature being excluded since it corresponded with the target variable.

# [Machine Learning Models](https://github.com/BOLTZZ/Novel-Deep-Learning-IDS-Code/tree/main/Machine%20Learning%20Models)
Machine learning models were created to achieve basline accuracies to compare against the deep learning models and the final deep learning algorithm. Z-score normalization was perfomed before implementing the machine learning models. These models were trained on 3 different types of data categories: only numerical features, only categorical features, or categorical and numerical features. Confusion matricies were printed out for each model to get a sense of the model's performance.

<i>Logistic Regression:</i>

One of the simplest machine learning models, logistic regression, was applied to the datasets. As expected, logistic regression had the lowest accuracies out of all the models. Different solvers were utilized to maximize accuracy.

<i>Decision Tree:</i>

Decision trees performed better than the logistic regression models. 

<i>Random Forest:</i>

Random forest models were trained with 70 trees. Random forests performed the best out of all the machine learning models.

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

# [Final Deep Learning Algortihm](https://github.com/BOLTZZ/Deep-Learning-IDS-Code/blob/main/Deep%20Learning%20Models/Deep%20Learning%20Algorithm.ipynb)

A deep learning algorithm was created based of the CNN model since it performed the best. This algorithm only utilized the 9 features selected from the feature selection/feature comparision. Label encoding was performed on the categorical features to preserve the columns for the architecture of the model. This model was trained for 10 epochs with a batch size of 64. The accuracy and loss were obtained to evaluate the model.

# Visualizing the Workflow
![Visual of the workflow described.](https://github.com/BOLTZZ/Deep-Learning-IDS-Code/blob/main/Images/IDS%20workflow.png)

# Table of Results
<table>
  <tr>
   <td rowspan="2" ><em>Model:</em>
   </td>
   <td colspan="2" ><em>Accuracies:</em>
   </td>
  </tr>
  <tr>
   <td><em>UNSW-NB15 Data:</em>
   </td>
   <td><em>Kyoto 2006+ Data:</em>
   </td>
  </tr>
  <tr>
   <td>Linear Regression
   </td>
   <td>98.62%
   </td>
   <td>97.60%
   </td>
  </tr>
  <tr>
   <td>Decision Tree
   </td>
   <td>98.76%
   </td>
   <td>99.13%
   </td>
  </tr>
  <tr>
   <td>Random Forest
   </td>
   <td>99.05%
   </td>
   <td>99.14%
   </td>
  </tr>
  <tr>
   <td>Multi-Layer Perceptron
   </td>
   <td>99.79%
   </td>
   <td>99.99%
   </td>
  </tr>
  <tr>
   <td>Variational Autoencoder
   </td>
   <td>97.08%
   </td>
   <td>38.19%
   </td>
  </tr>
  <tr>
   <td>Convolutional Neural Network
   </td>
   <td>99.81%
   </td>
   <td>99.99%
   </td>
  </tr>
  <tr>
   <td>Deep Learning Algorithm
   </td>
   <td>99.42%
   </td>
   <td>99.54%
   </td>
  </tr>
</table>

While, the Deep Learning Algorithm had a lower accuracy than the CNN and MLP, it only used 9 generalizable features. On the other hand, all the other models used 23 features for Kyoto 2006+ and 47 features for UNSW-NB15. Therefore, the Deep Learning Algorithm had an extremely high accuracy while still being generalizable.

# Similar Features Used
<table>
  <tr>
   <td><em>UNSW-NB15:</em>
   </td>
   <td><em>Type of Feature for UNSW-NB15:</em>
   </td>
   <td><em>Kyoto 2006+:</em>
   </td>
   <td><em>Type of Feature for Kyoto 2006+:</em>
   </td>
   <td><em>Description:</em>
   </td>
  </tr>
  <tr>
   <td>srcip
   </td>
   <td>Categorical
   </td>
   <td>Source_IP_Address
   </td>
   <td>Categorical
   </td>
   <td>Source IP address.
   </td>
  </tr>
  <tr>
   <td>sport
   </td>
   <td>Numerical
   </td>
   <td>Source_Port_Number
   </td>
   <td>Numerical
   </td>
   <td>Source port number.
   </td>
  </tr>
  <tr>
   <td>dstip
   </td>
   <td>Categorical
   </td>
   <td>Destination_IP_Address
   </td>
   <td>Categorical 
   </td>
   <td>Destination IP address.
   </td>
  </tr>
  <tr>
   <td>dsport
   </td>
   <td>Numerical
   </td>
   <td>Destination_Port_Number
   </td>
   <td>Numerical
   </td>
   <td>Destination port number.
   </td>
  </tr>
  <tr>
   <td>proto
   </td>
   <td>Categorical
   </td>
   <td>Protocol
   </td>
   <td>Categorical
   </td>
   <td>Transaction protocol.
   </td>
  </tr>
  <tr>
   <td>dur
   </td>
   <td>Numerical
   </td>
   <td>Duration
   </td>
   <td>Numerical
   </td>
   <td>Total duration of connection.
   </td>
  </tr>
  <tr>
   <td>sbytes
   </td>
   <td>Numerical
   </td>
   <td>Source bytes
   </td>
   <td>Numerical
   </td>
   <td>Source to destination transaction bytes.
   </td>
  </tr>
  <tr>
   <td>dbytes
   </td>
   <td>Numerical
   </td>
   <td>Destination bytes
   </td>
   <td>Numerical
   </td>
   <td>Destination to source transaction bytes.
   </td>
  </tr>
  <tr>
   <td>service
   </td>
   <td>Categorical
   </td>
   <td>Service
   </td>
   <td>Categorical
   </td>
   <td>Connection’s service type (http, telnet, etc).
   </td>
  </tr>
</table>
