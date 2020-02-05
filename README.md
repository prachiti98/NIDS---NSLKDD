# BitEndsApplication
Machine Learning based IDS

A Network Intrusion Detection System (NIDS) trained on NSL-KDD+ dataset to detect DoS and Probe attacks using different Machine
Learning models/algorithms. Feature selection algorithms are used improve the performance of the model and classifier leading to more
efficient results.

<b>Feature Selection</b>

● The CfsSubsetEval algorithm performs a selection among the attributes in the dataset that those are highly related to the class and that are less
important. In this way, the most important features of the dataset areidentified. CfsSubsetEval method uses BestFirst search algorithm.

● The Wrapper algorithm generates multiple subsets from the NSL-KDD+ 20% dataset and uses different classification algorithms (Random Forest,
kNN, Gaussian Naive Bayes) to induce classifiers from features in eachsubset. It then selects the features with the best classifier

Real-time log analysis

We have simulated DoS and Scan attacks on our machine using Pentmenu tool and captured the packets using tcpdump.
We convert this captured tcpdump data into useful logs with appropriate features which can then be tested on our model to detect if
some attack has taken place or not.

References

● Unal Cavusoglu, “A new hybrid approach for intrusion detection using machine learning methods”
http://dx.doi.org/10.1007/s10489-018-01408-x

● Tcpdump to KDD’99
https://github.com/inigoperona/tcpdump2gureKDDCup99/

