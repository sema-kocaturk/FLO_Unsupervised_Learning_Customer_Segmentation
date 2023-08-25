# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 18:06:26 2023

@author: user
"""

import datetime as dt
import pandas as pd
from sklearn.preprocessing import  StandardScaler
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from scipy import stats
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


df_ = pd.read_csv(r"C:\Users\user\Desktop\Datas\flo_data_20k.csv")
df = df_.copy()

df.info()

df["first_order_date"] = df["first_order_date"].apply(pd.to_datetime)
df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)
df["last_order_date_online"] = df["last_order_date_online"].apply(pd.to_datetime)
df["last_order_date_offline"] = df["last_order_date_offline"].apply(pd.to_datetime)


df["total_num_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value_order"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################


df["last_order_date"].max()
df.columns
today_date = dt.datetime(2021, 6, 1)


df["customer_id"] = df["master_id"]
df["recency_weekly"] = ((df["last_order_date"] - df["first_order_date"])/7).astype('timedelta64[D]')
df["T_weekly"] = ((today_date - df["first_order_date"])/7).astype('timedelta64[D]')


cltv_df= df[["order_num_total_ever_offline","order_num_total_ever_online",
             "customer_value_total_ever_offline","customer_value_total_ever_online",
             "recency_weekly","T_weekly"]]


# 1. Değişkenleri standartlaştırınız.
#SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(cltv_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(cltv_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(cltv_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(cltv_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(cltv_df,'recency_weekly')
plt.subplot(6, 1, 6)
check_skew(cltv_df,'T_weekly')
plt.tight_layout()
plt.savefig('after_transform.png', format='png', dpi=1000)
plt.show(block=True)



# Normal dağılımın sağlanması için Log transformation uygulanması
cltv_df['order_num_total_ever_online']=np.log1p(cltv_df['order_num_total_ever_online'])
cltv_df['order_num_total_ever_offline']=np.log1p(cltv_df['order_num_total_ever_offline'])
cltv_df['customer_value_total_ever_offline']=np.log1p(cltv_df['customer_value_total_ever_offline'])
cltv_df['customer_value_total_ever_online']=np.log1p(cltv_df['customer_value_total_ever_online'])
cltv_df['recency_weekly']=np.log1p(cltv_df['recency_weekly'])
cltv_df['T_weekly']=np.log1p(cltv_df['T_weekly'])
cltv_df.head() 

num_col=[]

for col in cltv_df.columns:
    if cltv_df[col].dtype != "O":
       num_col.append(col)


def standart_scaler(dataframe,col_name):
    ss = StandardScaler()
    dataframe[col_name] = ss.fit_transform(dataframe[[col_name]])
    cltv_df.head()
for i in num_col:
    standart_scaler(cltv_df,i)
     
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))   
elbow.fit(cltv_df)
elbow.show()

elbow.elbow_value_


#  Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(cltv_df)
kmeans.n_clusters
kmeans.cluster_centers_
segments = kmeans.labels_
cltv_df[0:5]


final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online","recency_weekly","T_weekly"]]

final_df["segment"] = segments
final_df.head()
final_df["segment"].value_counts()


#  Herbir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency_weekly":["mean","min","max"],
                                  "T_weekly":["mean","min","max","count"]})


hc_complete = linkage(cltv_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)


#  Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=7)
segments = hc.fit_predict(cltv_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline"
               ,"customer_value_total_ever_online","recency_weekly","T_weekly"]]
final_df["segment_hierarchical"] = segments
final_df.head()

#  Herbir segmenti istatistiksel olarak inceleyeniz.

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency_weekly":["mean","min","max"],
                                  "T_weekly":["mean","min","max","count"]})


hc_complete = linkage(cltv_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)