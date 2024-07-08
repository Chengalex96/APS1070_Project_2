# APS1070_Project_2: Anomaly Detection Algorithm using Gaussian Mixture  Model
 
**Key Takeaways:**
-	Large cases of transactions, small amount of fraud, unbalanced data
 -	If you predict everything as normal, it would still look good
-	Split the data to train and test
-	Plot distribution for density vs feature for fraud and normal transactions using matplotlib and histplot for bins
-	Used for a Gaussian model, to retrieve the mean and std. We want the means of normal to be different from the means of fraud and also like a small std so the curves don’t overlap
-	Fit single Gaussian distribution on a single feature using Sklearn.GasussianMixture, calculate the Area under the curve using the sklearn.score module
o	Choosing the best feature to fit the whole dataset, we calculate the F1 score with sklearn.metrics.f1_score
o	Find the best parameters for mean/threshold for max F1 score
-	Fitting multiple features onto one Gaussian by visually determining if there’s a relation between two features for normal vs fraud. Determine how many clusters there are, try to fit the fraud transactions onto a cluster
-	Fit a single feature with two Gaussian distributions, one for normal and one for fraud, and calculate the max thresholds for both that provide the best F1_score
-	Multiple Gaussians, multiple features: Two Gaussian models, multiple features, single feature for valid, various components for frauds since they happen in more than 1 cluster. There is an optimal number of features to maximize F1 score. 

**Code Summary:**

**Part 1: Getting Started**
- Working with a credit card fraud dataset, which contains 28 key features that aren't interpretable
- The csv file is converted to a pandas dataframe (285k rows)
 - df = pd.read_csv('creditcard.csv')
-  A column target that represents the fraud transactions (1), based on the fact there are so many more 0's (Normal)
-  Unbalanced dataset, there are 99.82% normal transactions and 0.1727% fraud transactions
 - Count: Valid = np.sum(df['Class'] ==0), Fraud = np.sum(df['Class'] ==1)

Split the dataset into 70% training and 30% test data:

 ![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/edc5f58d-29c1-4fe5-8123-17820bce5d30)

 Examine the difference in distribution between each feature for fraud and normal transactions:
 
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/3e1240d2-4d81-40b1-9840-f973cae19a85)

- Only the features that contain the letter 'V', ignore the time and target column, plot a histogram for each feature according to normal and fraud transactions

 ![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/69997828-dc1c-4b3c-b3e7-ceb4e5b6bb34)

 The graphs help with anomaly detection by providing us with the mean and standard deviation. Want the means to be largely distinct to classify from one another and small standard deviation to avoid overlap.

** Part 2: Single Feature Model with One Gaussian Distribution**
- Fitting Single Gaussian Distribution on a single feature of the full training dataset
 - from sklearn.mixture import GaussianMixture, gm = GaussianMixture(n_components = 1, covariance_type = 'full', random_state=0, ), gm.fit(X_train['V1'].values.reshape(-1,1))

- Calculate the Area under the Receiver Operator Curve (ROC)(AUC)
 - from sklearn.metrics import roc_auc_score, p = gm.score_samples(X_train['V1'].values.reshape(-1,1)), roc_auc_score(y_train, 1-p) 

- Calculate AUC for all features to distinguish the best feature to distinguish fraud transactions
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/cf6b3cdd-7b3b-4fe4-ace5-9694e64e7b23)
- print(AUC_total.max()), print(AUC_total.argmax()), the AUCs are placed in a pandas dataframe

Fit Feature 14 to a Gaussian model to find p which is then sorted

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/05ede338-5fcf-425f-b927-f8405bfee1ab)

Calculating the optimal threshold that provides us with the maximum F1_Score

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/14f21566-e639-406a-9200-53c5ba18d698)

The optimal threshold is -20.06 for a max F1_score of 0.61422

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/f04e134c-b075-4f6f-8029-1410e77de6ca)

If we fit the Gaussian model only to the normal transactions:

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/4b73d157-13ac-4c8e-a7e9-b1975f9de478)
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/f617de86-1220-41f0-9b21-87ebda344b51)

Didn't make a difference since there are so few fraud transactions relatively

**Part 3: Multiple features with one Gaussian Distribution**

- Two features are randomly selected and plotted on a scatter plot, labeled as fraud and non-fraud

 ![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/59c530d8-3c4c-4cb0-9073-6b6c3deaf603)
 ![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/4025575c-514a-411b-b700-b19d6ebd20f0)

 - Can choose the two features (V14 and V17) that provided the highest AUC to provide a threshold to maximize the F1_score
   
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/d18902c2-6719-4799-945d-c2fe8602a2e5)

The optimal threshold is -19.3833 with a maximum F1_score of 0.6864 - we do see improvement to the F1_Score compared to just 1 Gaussian component

Plotting the two features:

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/476a0185-9204-40f6-b1c0-fa08e8e4dbc4)
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/79530266-55a5-411d-a380-0d898d0727e1)

**Part 4: Single Feature with two Gaussian distributions**
- One distribution is for non-frauds, and the second is the fraud
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/cc94ca65-9858-4a43-83e9-7a073fff639b)

Calculate the Score Samples for G1 and G2

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/9e4115a9-bb2d-4029-9718-4bbed5d75a03)

Want to find an optimal c that maximizes the F1 score such that if S1<c*S2, the classification is a fraud

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/a1fabfba-b60e-4992-9fae-ccd3b6b6c3fa)
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/f096391f-cd6f-41fd-9c80-a2dc8ab95da2)

Used a for loop to determine the best feature that provides the highest F1_score:

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/3cfd3729-cf30-488b-af3b-3bc474218e1b)

Found the threshold for feature 12:

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/bc85a4ce-5650-404e-8b91-399001cdcafa)
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/7b8a6ef2-fbb6-4258-92e5-051cad5290df) 

Part 5: Multivariate and Mixture of Gaussian Distribution
- Determine the number of features and number of Gaussian distributions to use

- Model 1: Supervised (Fit between normal and frauds) Two Gaussian Models, Only with Features > AUC > 0.9 (5 Features), 1 Gaussian Component

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/1f8320d3-a7a6-4529-b1f0-6c25dee0acc7)
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/8f3942e2-222c-4737-8ca4-467731247c42)
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/8d6f2849-d20d-46df-957a-75aed21b6363)

- Model 2: Supervised (Fit between normal and fraud) Two Gaussian models, Features above AUC>0.9 (5 features), 3 Gaussian component

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/eca556d6-1e6b-490c-aaa6-c4b609e6fd8d)

We can observe that putting more Gaussian components helped improve the F1_Score

- Model 3 - Supervised (Fit between normal and fraud) Two Gaussian Models, Top 3 AUC Features, 3 Gaussian components - Try fewer features

- Model 4 - Supervised (Fit between normal and fraud) Two Gaussian models, Features above 0.8 AUC, 3 Gaussian components - Try more features

- Model 5 - Supervised (Fit between normal and fraud) Two Gaussian Models, Features above 0.9 AUC, 1 Gaussian component for normal transactions, 3 Gaussian components for fraud transactions

We can see that splitting up the normal and fraud transactions into different amounts of Gaussians components improves the F1_Score

- Model 6 - Supervised (Fit between normal and fraud) Two Gaussian Models, Features above 0.85 AUC,  1 Gaussian component for normal transactions, 3 Gaussian components for fraud transactions, Split Gaussian component works - try adding more features

- Model 7 - Unsupervised (Data has normal and fraud) One Gaussian model, Features above 0.85 AUC, 1 Gaussian component for all transactions.
 - Try unsupervised (not separating data) - more features have lower F1_score than only 1 feature (Part 2)

 - We observe that without filtering the fraud and normal transactions, we have only 1 Gaussian model which leads to a lower F1_Score

- Model 8 - Unsupervised (Data has normal and fraud) One Gaussian Model, Features above 0.85 AUC, 3 Gaussian component
 - Adding Gaussian components significantly makes the F1_Score lower for single Gaussian models

- Model 9 - Supervised (Fit between normal and fraud) Two Gaussian Models, Features above 0.80 AUC,  1 Gaussian component for normal transactions, 3 Gaussian components for fraud transactions
- It seems that unsupervised data and adding more Gaussian components don't help improve the F1_Score- We will stick to supervised data - 2 Gaussian models - Try different amounts of features

- Model 10 - Supervised (Fit between normal and fraud) Two Gaussian Models, Features above 0.85 AUC,  1 Gaussian component for normal transactions, 4 Gaussian components for fraud transactions, Try changing the number of Gaussian components

Model Summary:

![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/773e2516-6942-43e5-ad4b-8e3f7034e560)

Based on Validation Data: 
- It seems that unsupervised data doesn't work well since you won't have 2 Gaussian models to fit the data onto.
- There is an optimal number of features to maximize F1_Score. The features used were the features with an AUC > 0.85 - Features 3, 4, 7, 10, 11, 12, 14, 16, and 17
The number of Gaussian components is important for the fraud transactions since the trend for the fraud transactions seems to happen in more than 1 cluster. Requires more than 1 Gaussian distribution to fit the fraud transactions.
- Model 10- Supervised learning with 2 Gaussian models (for normal and fraud transactions), with the features above 0.85 AUC used (9 features), and 2 Gaussian components for the fraud transactions, 1 for the normal transactions provides the best F1_score of 0.8558

Part 6: Evaluating Performance on Test Set:
![image](https://github.com/Chengalex96/APS1070_Project_2/assets/81919159/562c00cf-0b13-4ef3-8ba9-595bf687d472)

Model 10 had the highest F1_Score for the validation data - therefore we will use that model for the test data.
- F1_Score: 0.8169
- Recall: 0.7891
- Precision: 0.8467

