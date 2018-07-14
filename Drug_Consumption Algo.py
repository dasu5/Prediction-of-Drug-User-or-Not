import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing

# declaring the data set file name
file_name = "drug_consumption.data"
# loading the data set
dataset = np.genfromtxt(file_name, delimiter=',', dtype='str')

data = dataset[:, 1:13]
target = dataset[:, 30]

attributes = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore',
              'Cscore', 'Impulsive', 'SS']

target_names = ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']

#pre-process the data using Label Encoder
LEncoder = preprocessing.LabelEncoder()
LEncoder.fit(target_names)
process_Target = LEncoder.transform(target)

X = data
y = process_Target

#splitting the data set in to train set and test set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20)

#calling the decision tree classifier
classifier = DecisionTreeClassifier(criterion="gini", splitter='random', max_leaf_nodes=10, min_samples_leaf=5,
                                  max_depth=5)

#fitting the training data to the classifier
classifier.fit(XTrain, yTrain)

prediction = classifier.predict(XTest)

print(data.shape)
print("Accuracy : ", (accuracy_score(yTest, prediction) * 100))
print("Confusion Matrix")
print(confusion_matrix(yTest, prediction))
print("Classification Report")
print(classification_report(yTest, prediction))

# decision tree visualization using graphviz and pydotplus
dot_data = StringIO()
tree.export_graphviz(classifier,
                     out_file=dot_data,
                     feature_names=attributes,
                     class_names=target_names,
                     filled=True, rounded=True,
                     impurity=False, special_characters=True)

#exporting the tree in to a PDF
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("DrugConsumption.pdf")
