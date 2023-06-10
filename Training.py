import pickle
f=open("./data.pickle","rb")
dict=pickle.load(f)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

data=np.array(dict['data'])
labels=np.array(dict['labels'])

X_train , X_test , Y_train ,Y_test = train_test_split(data ,labels ,test_size=0.2,shuffle=True,stratify=labels)
#print(X_train.shape)
#rint(X_test.shape)


model=RandomForestClassifier()
model.fit(X_train , Y_train)
Y_predict=model.predict(X_test)

score=accuracy_score(Y_predict,Y_test)

print("Accuracy Score :{}".format(score*100))

f=open('model.p',"wb")
pickle.dump({"model":model},f)
f.close()
