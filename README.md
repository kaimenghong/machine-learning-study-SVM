# machine-learning-study-SVM
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandarScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
digits= load_digits()
digits.data.shape
x_train,x_text,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
y_train.shape
ss=StandarScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
lsvc=LinearSVC()
lsvc.fit(x_train,y_train)
y_predict=lsvc.predict(x_test)
print 'The Accuracy of Linear SVC is', lsvc.score(x_test,y_test)
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))
