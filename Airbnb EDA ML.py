import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

pd.set_option('display.max_rows',100)

df = pd.read_csv("listings.csv.gz")
#print(df.head())
#print(df.columns)
#print(df.describe())
#print(df.info())
#print(df.shape)

print(df.isna().sum().sort_values(ascending=False).head(20))
print((df.isnull().sum() / len(df) *100).sort_values(ascending=False).head(20))

close_to_check = ['neighbourhood', 'neighborhood_overview','license', 'host_about' , 'host_location']
#for col in close_to_check:
#    print(f"\n\n--- {col} ---")
#    print(df[col].head(10))

cols_to_drop = ['id', 'listing_url', 'scrape_id', 'host_id', 'host_url',
 'host_neighbourhood','description', 'host_name', 'picture_url', 'host_thumbnail_url',
 'first_review', 'last_review', 'calendar_last_scraped',
 'availability_30', 'availability_60', 'availability_90', 'availability_365',
 'minimum_minimum_nights', 'maximum_minimum_nights',
 'minimum_maximum_nights', 'maximum_maximum_nights',
 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
 'last_scraped','source','name', 'host_since','host_picture_url', 'host_verifications',
 'host_has_profile_pic','has_availability','instant_bookable',
 'neighbourhood', 'neighborhood_overview','license', 'host_about' ,
 'host_location','calendar_updated','neighbourhood_group_cleansed']
df.drop(columns= cols_to_drop, inplace=True)

print(df.isna().sum().sort_values(ascending=False).head(80).reset_index())
print((df.isnull().sum() / len(df) *100).sort_values(ascending=False).head(80).reset_index())

num_cols = df.select_dtypes(include=['Int64','Float64']).columns
for col in num_cols:
    df[col]= df[col].fillna(df[col].median())
    
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols :
    df[col] = df[col].fillna('Unknown')

df['price'] = df['price'].replace('Unknown',None)
    
print((df.isna().sum()/len(df)*100).sort_values(ascending=False).reset_index())

obj_cols = df.select_dtypes(include='object').columns
print(obj_cols)             

df['price'] = df['price'].str.replace('$','',regex=False).str.replace(',','').astype(float)

cols_to_encode = [
    'host_response_time', 'host_response_rate', 'host_acceptance_rate',
    'host_is_superhost', 'host_identity_verified', 'neighbourhood_cleansed',
    'property_type', 'room_type', 'bathrooms_text', 'amenities', 'price'
]
le = LabelEncoder()
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col].astype(str))

x= df.drop(columns=['price'])
y= df['price']

from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split( x , y ,test_size=0.2 , random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error

model = RandomForestRegressor(n_estimators=200 , random_state=42 , n_jobs=-1)
model.fit(X_train , Y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test ,y_pred)
mse = mean_squared_error(Y_test , y_pred)
rmse = mse ** 0.5

print("MAE:",mae)
print("RMSE:",rmse)
print("Model score:",model.score(X_test , Y_test))
