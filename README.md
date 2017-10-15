# NLP_Maximum_Entropy_Classifier
Trains on test data from IMDB movie reviews, and classifies future documents based on sentiment.
Please see "Maxent and Perceptron.pdf" for detailed implementation notes.

An ongoing project of mine in this repository is the empirical model of the MaxEnt perceptron. If you're curious, check out MaxEnt_Empirical_Notes_Model. I've written the notes and the code following this paper:

http://blog.datumbox.com/machine-learning-tutorial-the-max-entropy-text-classifier/

The maximum entropy classifer I have successfully implemented utilizes stochastic gradient descent, whereas the empirical model utilizes regular gradient descent. This was a design choice because stochastic descent is more of an approximate convergence, whereas the empirical model is more exact based upon frequency probability. I wanted the program to be able to run, train, and score within a decent amount of time. Currently, the program is still a bit slow because of how I've handled the Bag of Words, but still gives a 74-78% accurracy across 10 test splits on average. The theoretical maximum is 86.7 percent accuracy, but that is with much more in-depth feature extraction, whereas I am essentially utilizing Bag of Words and removing stopwords in scoring.

Enjoy!
