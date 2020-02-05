# BitEndsApplication
<b><i>Machine Learning based IDS</i></b>

A Network Intrusion Detection System (NIDS) trained on NSL-KDD+ dataset to detect DoS and Probe attacks using different Machine
Learning models/algorithms. Feature selection algorithms are used improve the performance of the model and classifier leading to more
efficient results.

Install the requirements by:
pip install -r requirements.py (Make sure your default is Python 3.6+)

<b>Feature Selection</b>

● The CfsSubsetEval algorithm performs a selection among the attributes in the dataset that those are highly related to the class and that are less
important. In this way, the most important features of the dataset areidentified. CfsSubsetEval method uses BestFirst search algorithm.

● The Wrapper algorithm generates multiple subsets from the NSL-KDD+ 20% dataset and uses different classification algorithms (Random Forest,
kNN, Gaussian Naive Bayes) to induce classifiers from features in eachsubset. It then selects the features with the best classifier

To get the results run it like:
python wrapper.py

<b>Real-time log analysis</b>

We have simulated DoS and Scan attacks on our machine using Pentmenu tool and captured the packets using tcpdump.
We convert this captured tcpdump data into useful logs with appropriate features which can then be tested on our model to detect if some attack has taken place or not.

CSVs got after the attack:
UDPScan.csv
Slowloris.csv
QuickScan.csv

To get the results run it like:
python try.py

We compare the accuracy values obtained from different classifiers such as <b>Random Forest, kNN, SVM and Ensemble technique<b>.
Accuracy values are calculated from the obtained confusion matrix as well as by using 10 fold cross validation technique insome cases.

To run the model:
python index.py

<b>References</b>

● Unal Cavusoglu, “A new hybrid approach for intrusion detection using machine learning methods”
http://dx.doi.org/10.1007/s10489-018-01408-x

● Tcpdump to KDD’99
https://github.com/inigoperona/tcpdump2gureKDDCup99/

