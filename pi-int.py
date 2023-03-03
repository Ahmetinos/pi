import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)


pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

df=pd.read_csv("Datasets//dataset.csv")

print("#############Part:1-> First Look at the Dataset ##############"
      "\n")
def check_df(dataframe,head=5,tail=5):
    print("############# Shape #############")
    print(dataframe.shape)
    print("############# Types #############")
    print(dataframe.dtypes)
    print("############# Head #############")
    print(dataframe.head(head))
    print("############# Tail #############")
    print(dataframe.tail(tail))
    print("############# NA #############")
    print(dataframe.isnull().sum())
    print("############# Quantiles #############")
    print(dataframe.describe([.05,.25,.5,.75,.95]).T)

check_df(df)

print("\n")

print("####### Part:2 -> Data Visualization-1 ##################### ")

def var_evaluation_bool(database,bool_var="isVirus",desc=False,barplot=False):
    if  desc:
        print(database[database[bool_var]==True].describe([.05,.25,.5,.75,.95]).T)
        print(database[database[bool_var]==False].describe([.05,.25,.5,.75,.95]).T)
    if barplot:
        df[bool_var].value_counts().plot(kind="bar")
        return plt.show()

print("\n")
var_evaluation_bool(df,bool_var="isVirus",desc=True) #Statistical description
#feature_1->isVirus=False values are higher.
#feature_2->Although a clear distinction cannot be made, low values are usually isVirus=True values.
#feature_3->Extreme values (too high or too low) are usually close to each other when isVirus=True.
#feature_4->High values are generally in isVirus=False.
#var_evaluation_bool(df,bool_var="isVirus",barplot=True) #Barplot
#The number of False values is about twice the number of True values.

print("\n")

def var_evaluation_con(database,con_var1=None,con_var2=None,hue="isVirus",scatterplot=False):
    if (con_var1==None) & (con_var2==None):
        sns.pairplot(database, hue=hue)
        return plt.show()
    elif(con_var1!=None) & (con_var2==None):
        sns.boxplot(data=database,x=con_var1)
        return plt.show()
    elif(con_var1==None) & (con_var2!=None):
        sns.boxplot(data=database,x=con_var2)
        return plt.show()
    else:
        if scatterplot:
            sns.scatterplot(data=database, x=con_var1, y=con_var2, hue=hue)
        return plt.show()

var_evaluation_con(df)
#var_evaluation_con(df,con_varl1="feature_2")
#var_evaluation_con(df,con_var2="feature_3")
#var_evaluation_con(df,con_var1="feature_1",con_var2="feature_3",scatterplot=True)


#As far as I can see from the boxplots, feature_2 and feature_3 have outliers.
# I will do outlier analysis for 4 features and edit these values.

print("\n")

print("####### Part:3-> Outliers ##################### ")
def outlier_thresholds(database,var,q1_range=.1,q3_range=.9):
    q1=database[var].quantile(q1_range)
    q3=database[var].quantile(q3_range)
    iqr=q3-q1
    low=q1-1.5*iqr
    up=q3+1.5*iqr
    return low,up
def check_outlier(database,var):
    low,up=outlier_thresholds(database,var)
    return database[(database[var]<low) | (database[var]>up)].any(axis=None)

outlier_columns=[]
for col in df.columns:
    print(col,check_outlier(df,col))
    if check_outlier(df,col)==True:
        outlier_columns.append(col)

print("\n")

def grab_outliers(database,var,index=False):
    low,up=outlier_thresholds(database,var)
    if database[(database[var]<low) | (database[var]>up)].shape[0]>10:
        print(database[(database[var]<low) | (database[var]>up)].head())
    else:
        print(database[(database[var]<low) | (database[var]>up)])
    if index:
        outlier_index=database[(database[var]<low) | (database[var]>up)].index
        return outlier_index

for col in outlier_columns:
    grab_outliers(df,col) #For feature_2->Index-24 and For feature_3->Index-679

print("\n")

#Although deleting was a solution, it would result in data loss.
# Since we do not have a large amount of data, I considered the "suppression" method appropriate.

def replace_with_thresholds(database,var):
    low,up=outlier_thresholds(database,var)
    database.loc[database[var]<low,var]=low
    database.loc[database[var]>up,var]=up

for col in outlier_columns:
    replace_with_thresholds(df,col)

print(df.iloc[24])
#print(df.iloc[679])

print("\n")

print("####### Part:4-> Missing Values ##################### ")

print("\n")

print(df[df["isVirus"]==True]["feature_1"].mean())
features_cols=df.columns[:-1]

print("\n")

def miss_values(database):
    for feature in features_cols:
        database.loc[database["isVirus"] == True,feature ] = \
            database[feature].fillna(database[database["isVirus"]==True][feature].mean(),axis=0)
        database.loc[database["isVirus"] == False,feature ] = \
            database[feature].fillna(database[database["isVirus"]==False][feature].mean(),axis=0)


miss_values(df)

print("\n")

print(df[df["isVirus"]==True]["feature_1"].mean())

print("\n")
#The averages will not change because they are populated with average values.


print("####### Part:5-> Data Visualization-2 ##################### ")

print("\n")

var_evaluation_con(df)

#var_evaluation_con(df,con_var1="feature_2") #Quantiles of this graph->0.25 and 0.75
#var_evaluation_con(df,con_var2="feature_3") #Quantiles of this graph->0.25 and 0.75


print("\n")

print("####### Part:6-> Machine Learning ##################### ")

print("\n")

X=df.drop("isVirus",axis=1)
Y=df["isVirus"]

def heatmap(database):
    sns.heatmap(database, annot=True, fmt='.2f', cmap=sns.diverging_palette(220, 20, n=15, as_cmap=True),
                       linewidths=1)
    return plt.show()

heatmap(df.corr())

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7,stratify=Y,random_state=213)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.6,stratify=y_train,random_state=4)
print(X_train.shape[0],X_valid.shape[0],X_test.shape[0])

print("\n")

##Standardization
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_valid=scaler.fit_transform(X_valid)
X_test=scaler.fit_transform(X_test)

##MultiLayer Perceptron Classifier
mlpc=MLPClassifier(max_iter=1000)
mlpc_model=mlpc.fit(X_train,y_train)
y_pred=mlpc_model.predict(X_test)
print(f1_score(y_test,y_pred))

print("\n")

###Model Tuning
mlpc_params={"alpha":[0.1,0.02,0.01,0.005,0.001,0.0001],"hidden_layer_sizes":[(10,10,10),(100,100,100),(100,100),(3,5),(5,3)],
            "solver":["lbfgs","adam","sgd"],"activation":["relu","logistic"]}
mlpc=MLPClassifier(max_iter=500)
mlpc_cv=GridSearchCV(mlpc,
                     mlpc_params,
                     cv=5,
                     n_jobs=-1)
mlpc_cv_model=mlpc_cv.fit(X_valid,y_valid)

print("The best parameters:" +str(mlpc_cv_model.best_params_))

print("\n")

###Re_modelling
mlpc=MLPClassifier(alpha=0.1,activation="relu",hidden_layer_sizes=(100,100,100),solver="adam",max_iter=1000)
mlpc_tuned=mlpc.fit(X_train,y_train)
y_pred=mlpc_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))

heatmap(confusion_matrix(y_test,y_pred))

##CONCLUSION:
#After tuning the model, I increased the success from 75-80% to around 87-88%.
#To refine the model:
#1) One of the two variables (existing between feature-1 and feature-4) with high similarity in the correlation matrix can be removed from the dataset.
#2) Trials can be made using PCA.
#3) Another algorithm (knn, decision tree etc.) can be used. Deep learning can be applied.