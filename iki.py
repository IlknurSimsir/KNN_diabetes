import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from itertools import accumulate
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import joblib
import warnings
 
#uyarıları kapat
warnings.filterwarnings('ignore')
sns.set_context("notebook")
sns.set_style("white")

def diabetes_to_numeric(diabetes_values):
    if diabetes_values=="Diabetes":
        return 1
    else:
        return 0 

def calculate_proportion(frequency_table,total_count):
    proportions=[]
    for value in frequency_table:
        proportions.append(value/total_count)
    return proportions

#veriyi yükleme 
df=pd.read_excel('Diabet_Siniflandirma.xlsx')
df.head()

#gereksiz sütunları kaldırma
df.drop(columns=['Unnamed: 16','Unnamed: 17','Patient number'],inplace=True)
print('Guncel sütunlar')
print(df.columns)

#hedef değişkenimizi (Diabetes) sayısal değere dönüştürme
df['Diabetes']=df['Diabetes'].apply(diabetes_to_numeric)#yukarıda yazılan fonksiyonu uyguladık

#sınıf dağılımını kontrol etme
frequency_table=df['Diabetes'].value_counts()
props=calculate_proportion(frequency_table, len(df['Diabetes']))
print("Sınıf dağılımı: ")
for cls,proportion in zip(frequency_table.index,props):
    print(f"class {cls} : {proportion :.2%}")


#kullanılacak sütunların seçimi
df_reduced=df[['Diabetes','Cholesterol','Glucose',
               'BMI','Waist/hip ratio',
               'Chol/HDL ratio','HDL Chol','Systolic BP',
               'Diastolic BP','Weight']]


#sayısal sütunların seçimi
numerical_columns=df_reduced.iloc[:,1:]


#veriyi ölçekleme
scaler=StandardScaler()
scaled_features=scaler.fit_transform(numerical_columns)

#ölçeklendirilmis dataframe dönüştürme
df_standardized=pd.DataFrame(scaled_features,columns=numerical_columns.columns)

#hedef değişkeni ekleme
df_stdize=pd.concat([df_reduced['Diabetes'],df_standardized],axis=1)
print("Standartlışılmış veri seti")
print(df_stdize.head())

X=df_stdize.drop(columns=['Diabetes'])
y=df_stdize['Diabetes']

X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=42)
print(f"Eğitim seti boyutu:{X_train.shape}")
print(f"Test seti boyutu:{X_test.shape}")

#KNN modeli oluşturma
knn =KNeighborsClassifier()

#modeli eğitme
knn.fit(X_train,y_train)

#test setinde tahmin yapma
y_pred=knn.predict(X_test)

#model doğruluğunu hesaplama
accuracy=accuracy_score(y_test, y_pred)
print(f"Model doğruluğu : {accuracy:.2%}")

#hiperparametre arama
param_grid={'n_neighbors':range(1,20)}
grid_search = GridSearchCV(knn, param_grid,cv=5)
grid_search.fit(X_train,y_train)
print(f"en iyi parametreler: {grid_search.best_params_}")
print(f"en iyi doğruluk oranı: {grid_search.best_score_:.2%}")

#karışıklık matrisi
cm=confusion_matrix(y_test, y_pred)
print(f"Karışıklık matrisi:\n{cm}")


#modeli kaydetme
model_file="knn_diabetes_model.pkl"
joblib.dump(knn,model_file)
print(f"Scaler: '{model_file}'olarak kaydedildi")

#Scaler olarak kaydetme
scaler_file="scaler.pkl"
joblib.dump(scaler,scaler_file)
print(f"Scaler: '{scaler_file}'olarak kaydedildi")


#Modeli yükle ve gerçek verilerle test et
model_file="knn_diabetes_model.pkl"
scaler_file="scaler.pkl"
model=joblib.load(model_file)
scaler=joblib.load(scaler_file)

#kullanıcıdan veri alma
print("Lütfen aşağıdaki değerleri girin")

cholesterol = float(input("Cholesterol: "))

glucose = float(input("Glucose: "))

bmi = float(input("BMI: "))

waist_hip_ratio = float(input("Waist/Hip Ratio: "))

hdl_chol = float(input("HDL Chol: "))

chol_hdl_ratio = float(input("Chol/HDL Ratio: "))

systolic_bp = float(input("Systolic BP: "))

diastolic_bp = float(input("Diastolic BP: "))

weight = float(input("Weight: "))

#Özellikleri bir araya getirme
features =np.array([[cholesterol,glucose,bmi,waist_hip_ratio,hdl_chol,chol_hdl_ratio,systolic_bp,diastolic_bp,weight]])

#özellikleri ölçeklendirme
scaled_features=scaler.transform(features)

#model ile tahmin yapma 
prediction= model.predict(scaled_features)

#tahmin sonucu gösterme
if prediction[0]==1:
    print("Diyabet")
else :
    print("Diyabet değil")



