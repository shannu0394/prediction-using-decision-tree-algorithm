# prediction-using-decision-tree-algorithm

Prediction using Decision Tree Algorithm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
Data overview
iris_df = pd.read_csv('Iris.csv')
iris_df.head()
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	1	5.1	3.5	1.4	0.2	Iris-setosa
1	2	4.9	3.0	1.4	0.2	Iris-setosa
2	3	4.7	3.2	1.3	0.2	Iris-setosa
3	4	4.6	3.1	1.5	0.2	Iris-setosa
4	5	5.0	3.6	1.4	0.2	Iris-setosa
iris_df = iris_df.drop('Id',axis=1)
iris_df.isna().sum()
SepalLengthCm    0
SepalWidthCm     0
PetalLengthCm    0
PetalWidthCm     0
Species          0
dtype: int64
Visualising the data
sns.set_style('darkgrid')
sns.pairplot(iris_df,hue='Species',palette='hls');

Preparing the dataset for training
x=iris_df.iloc[:,:-1]
y=iris_df.iloc[:,-1]
y.head()
0    Iris-setosa
1    Iris-setosa
2    Iris-setosa
3    Iris-setosa
4    Iris-setosa
Name: Species, dtype: object
x.head()
SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm
0	5.1	3.5	1.4	0.2
1	4.9	3.0	1.4	0.2
2	4.7	3.2	1.3	0.2
3	4.6	3.1	1.5	0.2
4	5.0	3.6	1.4	0.2
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
Training decision tree alogrithm
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('accuracy: ',accuracy_score(y_pred,y_test))
accuracy:  1.0
Visualising the decision tree
plt.figure(figsize=(50,30))
plt.style.use('grayscale')
plot_tree(classifier,filled=True,feature_names=x.columns);
