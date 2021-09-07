# Predicting pain based on Physiological data using Random Forests and Score-Level Fusion

Dakshin Rathan
1.
# Introduction

In this project, we are discussing a pain recognition system based on physiological data using score level fusion with random forest classifiers. This is an important development and accurately predicting whether someone is in pain is useful for soldiers in combat.

# Method


A decision tree is an algorithm that is able to classify input data based on a series of decisions. A basic structure of a decision tree is below. The tree starts at the root node, and makes splits, or decisions, where the data will move further down the tree. When a pure classification is achieved (ex. Pain or No Pain), the a decision is made.

![](RackMultipart20210904-4-35umen_html_95198936573ac3d8.jpg)

However, a single decision tree is vulnerable to overfitting to the training data and bias of a single data set. To comband this in this paper, we use random forest.

A random forest randomly samples a portion of the input data with replacement and creates many different and unique decision trees. Because of this, different splits occur and different decisions occur in each of the trees. After all the trees have made a decision, the random forest will make a classification in line with the majority decision among the trees.

![](RackMultipart20210904-4-35umen_html_3cdcb802ffbd1f0e.jpg)

In this project, we classify whether a subject is in pain or not based entries of signal data for four types of physiological signals: Respiration Rate (RES), Systolic Blood Pressure (SYS), Diastolic Blood Pressure (DIA), and electrodermal activity (EDA).

Training and Testing Random Forest

We used training and testing data of 30 subjects with 8 sets of physiological signal readings: 4 sets corresponding to each type of physiological signal when &quot;pain&quot; was reported and 4 sets when &quot;no pain&quot; was reported. To accomplish classification in this project, we trained four different random forests for each type of physiological signal. However, physiological signal entries are variable and random forests require each set of readings to contain the same number of features. To combat this, we downsampled the physiological readings of all entries to 5000 to create a uniform number of features. Then, we normalized each entry so values range between 0 and 1.

Because we are using one random forest for each type of physiological data, we filtered all of the signal readings for each type of physiological data for each random forest. For example, all of the RES training data will be filtered out to be trained on the RES random forest. This same process is done for the 3 other type of physiological data. The testing data is filtered out in the same way and is used as an input in the testing phase of the random forest. We generate a set of predictions for each subject in the testing data.

Score Level Fusion

After we get predictions from each random forest, we use a method called score level fusion to generate the final result for each subject. For each subject, we have a prediction from each of the 4 random forests corresponding to a type of physiological signal. Whichever choice (pain or no pain) has the majority vote among the 4 trees, is chosen as the final decision for a test subject.

Talk about the fusion approach you used. If you did the extra credit detail both of them.

1.
# Experimental design and results

We used two different csv files as splits for training and testing. We also printed the confusiong matrix, accuracy, recall and precision for each individual random forest and the combined prediction results after score level fusion for each alternative. The results of the experiment is below.

Data1.csv: testing

Data2.csv: training

![](RackMultipart20210904-4-35umen_html_cbec8e863ad8e05d.png)

![](RackMultipart20210904-4-35umen_html_b4a18237baf75ab2.png)

Data1.csv: training

Data2.csv: testing

![](RackMultipart20210904-4-35umen_html_5aee6ae6c04ce013.png)

![](RackMultipart20210904-4-35umen_html_d4ec5e4608849a2e.png)

We can sese that the accuracy, prescision and recall was generally higher when data1.csv was the testing data than vice versa.

1.
# Discussion and conclusion

I think physiological data is good for pain recognition. However, some physiological responses are shown to contribute to pain more so than others.

Do you think phsyiological data is good for pain recognition? What about fusion? Is there a better approach?

According to the data, the majority voting fusion approach did not improve the accuracy, recall or precision by very much. We think that fusion method from project 1, where the data was fused in the before random forest classification worked better. Though we cannot conclude anything definitively, we can see that by the results below from our project 1 algorithm using the project 1 fusion method with hand crafted features, that the method results in higher accuracy, recall, and precision.

![](RackMultipart20210904-4-35umen_html_be31549f3855d638.png)

We think that contributing multipole modalities can improve the accuracy of the machine learning models. In the papers we referenced in the jintroduction section, using data from thermal readings, fiacial photos, videos of different resolutions, and temporal and spatial analysis of those videos allowed for more reliable and accurate predictions. This allows the model to get a more full picture of the different variables that contribute to pain.
