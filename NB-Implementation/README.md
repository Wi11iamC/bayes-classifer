# Naive Bayes Implementation

### This project contains implementation of Naive Bayes classification algorithm. Naive Bayes classifier was implemented from scratch in `GaussianNaiveBayes.py` using the numpy package for the two datasets: (1) digits and (2) faces. The matplotlib package was used to create graphs, and charts of the collected data about the Naive Bayes model implemented in this project. 
<br>

---

<br>

#### The Naive Bayes Classifier is a probabilistic algorithm and based on Bayes' theorem. Baye's theorem states that the probability of a hypothesis (or label) is proportional to the probability of evidence (input feature space), given the hypothesis. Naive Bayes, assumes that the features are independent of each other given the class label. This asssumption simplify the computation of the likelihood of the evidence by using a simple product rule to compute the posterior probability of each class given the evidence.
<br>

---

<br>

### It includes the following directories and files:
- `main.py`: This is the main file of the project that can be run to execute the program.
- `results/`: This directory contains the data collected about the Naive Bayes model, such as training time, accuracy vs percentages, etc.
- `data/`: This directory contains the training, validation, and test data required for running the program.
- `data.zip`: This is the compressed version of the data/ directory.
<br>

---

<br>

## Running the Program
- To run the program, simply execute the main.py file.