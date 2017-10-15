import sys
import getopt
import os
import math
import operator

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds = 10

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """


    # Write code here

    return 'pos'
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
     *
     * Personal notes:
     *  1.) Let w_0,w_2,...w_n be words that exist in a document
     *  2.) Create a sparse array for every document
     *  3.) First, summarize the training set
     *      - To do this, create an array for document class, and one for the bag of words, and one for words.
     *      - x: In the document class, record 0 for negative class, and 1 for positive
     *      - y:ys: In the sparse array, record the occurence frequency of words/features.
     *      - N: In the word array, hold the size for the training dataset.
     *      - To summarize, ~p(x,y) = 1/N * frequency of (x,y) occurs in the sample
     *  4.) Next, apply the indicator function
     *      - f_j(x,y) = 1 if y = c_i and y contains w_k
     *                 = 0 otherwise
     *      - To help with this idea, the c_i is a positive match
     *      - Also, the w_k comes from every word in each document.
     *      - This is the creation of a feature.
     *  5.) Find the expected values of a feature
     *      - With respect to empirical distribution:
     *      - ~p(f_j) = sum_{x,y} ~p(x,y)f_j(x,y)
     *      - Which is lovely! We've just computed those two probabilities.
     *      - Compare:
     *      - Now, with respect to p(y|x), or probability of a class given relevant info
     *      - (this is to compare):
     *      - p(f_j)  = sum_{x,y} ~p(x)p(y|x)f_j(x,y)
     *      - where ~p(x) is the empirical distribution of the dataset, and is usually 1/N
     *      - Therefore, p(f_j) = ~p(f_j) = sum_{x,y} ~p(x,y)f_j(x,y)
     *      - NOTE: all '=' are logical equivalence.
     *  6.) Derive features from the trained data! How? Stochastic Gradient Descent!
     *
     *  TODO:
         - Find out how to implement bag of words.
         - Learn how to parse all input into their respective matrices
         - Put said matrices into the _init_(self) area for maxent
         - Utilize addExample to populate those matrices using the methods laid out
         - After gathering all of the data, calculate ~p(x,y) for each (x,y) (word in a document)
         - After that, calculate f_j(x,y) for each (x,y)
         - Utilizing the two calculated matrices, calculate ~p(f_j), or the probability of a class given relevant info
         - Once I've done that, I should be set up for my stochastic gradient work.
         -
         - Resource: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html  #bag of words
         - Resource: http://blog.datumbox.com/machine-learning-tutorial-the-max-entropy-text-classifier/ #maxent tutorial 
         - Resource: https://www.coursera.org/learn/ml-classification/lecture/DBTNt/l2-regularized-logistic-regression
         - Resource: https://msdn.microsoft.com/en-us/magazine/dn904675.aspx
    """

    # Write code here

    pass
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      *
      * Personal notes:
      *  The initial for-loop iterates through the examples given as training data.
      *  From what it seems right now, train is a complete function.
      """
      for example in split.train:
          words = example.words
          self.addExample(example.klass, words)
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
