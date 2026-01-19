import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression , SGDRegressor,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score



df=pd.read_csv("/home/kristo/house_web/Housing.csv")

X=df.drop('price',axis=1)
y=df['price']

num_cul=[col for col in X.columns if X[col].dtype!='object']
cat_cul=[col for col in X.columns if X[col].dtype=='object']




X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

imputer=SimpleImputer(strategy='median')

X_train_num_fill=imputer.fit_transform(X_train[num_cul])
X_test_num_fill=imputer.transform(X_test[num_cul])


X_train_cat_fill = SimpleImputer(strategy='most_frequent').fit_transform(X_train[cat_cul])
X_test_cat_fill = SimpleImputer(strategy='most_frequent').fit_transform(X_test[cat_cul])



encoder=OneHotEncoder(sparse_output=False,drop='first')
X_train_encoded=encoder.fit_transform(X_train_cat_fill)
X_test_encoded=encoder.transform(X_test_cat_fill)



standard=StandardScaler()
X_train_num_scaled=standard.fit_transform(X_train_num_fill)
X_test_num_scaled=standard.transform(X_test_num_fill)



X_train_final=np.concatenate((X_train_encoded,X_train_num_scaled),axis=1)
X_test_final=np.concatenate((X_test_encoded,X_test_num_scaled),axis=1)


LR_model=LinearRegression()
LR_model.fit(X_train_final,y_train)

scores=cross_val_score(LR_model,X_train_final,y_train,cv=5,scoring='neg_mean_squared_error')

mse_scores=-scores
rmse_scores=np.sqrt(mse_scores)

print('MSE',mse_scores)
print('RMSE',rmse_scores)
print('Avg_MSE',np.average(mse_scores))
print('Mean RMSE',rmse_scores.mean())
print('----------------------------')

r2_scores=cross_val_score(LR_model,X_train_final,y_train,cv=5,scoring='r2')

print('R2',r2_scores)
print('Avg_R2',np.average(r2_scores))