import os, cv2, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier

classes = ["Stage1","Stage2","Stage3"]
X,y=[],[]

def features(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return [np.mean(gray),np.std(gray),np.var(gray)]

for i,c in enumerate(classes):
    for f in os.listdir(f"dataset/{c}"):
        img=cv2.imread(f"dataset/{c}/"+f)
        if img is not None:
            img=cv2.resize(img,(256,256))
            X.append(features(img))
            y.append(i)

model=RandomForestClassifier()
model.fit(X,y)
joblib.dump(model,"model.pkl")
print("done")
