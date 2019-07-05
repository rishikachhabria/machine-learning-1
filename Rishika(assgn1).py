import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Final_Test=pd.read_excel('C:\Users\Acer\Desktop\Final_Test')
Final_Train=pd.read_excel('C:\Users\Acer\Desktop\Final_Train')

#Place
Final_Test['Place'].fillna("not-known",inplace=True)
Final_Test['Place']=Final_Test['Place'].str.split(",")
Final_Test['place']=Final_Test['Place'].str[0]
Final_Test['City']=Final_Test['Place'].str[-1]
Final_Test['City'].fillna("not-known",inplace=True)


Final_Train['Place'].fillna("not-known",inplace=True)
Final_Train['Place']=Final_Train['Place'].str.split(",")
Final_Train['place']=Final_Train['Place'].str[0]
Final_Train['City']=Final_Train['Place'].str[-1]
Final_Train['City'].fillna("not-known",inplace=True)


Final_Train['City'][3980]="not-known"
Final_Train['Place'][3980]="not-known"

Final_Test=pd.get_dummies(Final_Test,columns=["City","Profile"],prefix=["City","Profile"])
Final_Train=pd.get_dummies(Final_Train,columns=["City","Profile"],prefix=["City","Profile"])

#Qualification
q=list(Final_Test['Qualification'])
Final_Test['Qualification']=Final_Test['Qualification'].str.split(",")
qual_dict={}
for i in Final_Test['Qualification'].values:
    for j in i:
        j=j.strip()
        if j in qual_dict:
            qual_dict[j]+=1
        else:
            qual_dict[j]=1


qual=sorted(qual_dict.items(),key=lambda x:x[1],reverse=True)[:10]
final_qual=[]
for x in qual:
    final_qual.append(str(x[0]))
for y in final_qual:
    Final_Test[y]=0
for a,b in zip(Final_Test['Qualification'].values,np.array([idx for idx in range(len(Final_Test))])):
    for k in a:
        k=k.strip()
        if k in final_qual:
            Final_Test[k][b]=1


Final_Train['Qualification']=Final_Train['Qualification'].str.split(",")
qual_dict={}
for i in Final_Train['Qualification'].values:
    for j in i:
        j=j.strip()
        if j in qual_dict:
            qual_dict[j]+=1
        else:
            qual_dict[j]=1
        
qual=sorted(qual_dict.items(),key=lambda x:x[1],reverse=True)[:10]
final_qual=[]
for x in qual:
    final_qual.append(str(x[0]))
for y in final_qual:
    Final_Train[y]=0
for a,b in zip(Final_Train['Qualification'].values,np.array([idx for idx in range(len(Final_Train))])):
    for k in a:
        k=k.strip()
        if k in final_qual:
            Final_Train[k][b]=1


#Experience
Final_Test['Exp']=Final_Test['Experience'].str.slice(stop=2).astype(int)

Final_Train['Exp']=Final_Train['Experience'].str.slice(stop=2).astype(int)


#Rating
Final_Test['Rating'].fillna('0%',inplace=True)
Final_Test['rating']=Final_Test['Rating'].str.slice(stop=-1).astype(int)

Final_Train['Rating'].fillna('0%',inplace=True)
Final_Train['rating']=Final_Train['Rating'].str.slice(stop=-1).astype(int)

           

"""Final_Test.drop("Miscellaneous_Info",axis=1,inplace=True)
Final_Train.drop("Miscellaneous_Info",axis=1,inplace=True)"""

x_train=Final_Train.iloc[:,7:].values
y_train =Final_Train.iloc[:,5].values
  
x_test=Final_Test.iloc[:,6:].values 

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
y_train=sc_y.fit_transform(y_train)
y_train=y_train.reshape(-1,1)


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(sc_x.fit_transform(x_train),y_train)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x_train,y_train)



from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


y_pred=regressor.predict(x_test)

print regressor.score(x_train,y_train)
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(estimator=regressor,X=x_train,y=y_train,cv=10)
print (accuracy.mean())
print(accuracy.std())
