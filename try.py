import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
import os
import sys
import traceback
import pickle

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split



train_url = 'KDDTrain+.txt'
test_url = 'slowloris.csv'


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","extra"]


df = pd.read_csv(train_url,header=None, names = col_names)
df= df.drop(['extra'], axis=1)

df_test = pd.read_csv(test_url)

print(df_test.head(5))

print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)



selected_col_names = ['duration' ,'protocol_type', 'service', 'flag', 'src_bytes', 
                      'dst_bytes', 'wrong_fragment', 'hot', 'logged_in', 'land', 'urgent',
                      'count', "srv_count", "serror_rate", 
                      "rerror_rate", 'num_failed_logins', "diff_srv_rate", "dst_host_count", 
                      "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
                      "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
                      "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

for col in df_test.columns.tolist():
    if col not in selected_col_names:
        df_test.drop(col, axis=1, inplace=True)

df_test['service'] = 'http'
df_test['flag']    = 'S0'
df_test['label']   = 'normal'
df_test['urgent']  = 0
df_test['dst_host_count']  = 0
df_test['dst_host_srv_count']  = 0
df_test['dst_host_same_srv_rate']  = 0
df_test['dst_host_serror_rate'] = 0

print("Columns", df_test.columns.tolist())


def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame

df = select_columns(df, selected_col_names)
df_test = select_columns(df_test, selected_col_names)

print(df_test.head(1))

#Print Attribute Values
print('Label distribution Training set:')
print(df['label'].value_counts())
print()
print('Label distribution Test set:')
print(df_test['label'].value_counts())

print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())
print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


#Data Preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]

df_categorical_values.head()

# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
print(unique_protocol2)

# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
print(unique_service2)


# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
print(unique_flag2)



# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2


#do it for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2 

df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

print(df_categorical_values.head())
print('--------------------')
print(df_categorical_values_enc.head())

# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)


enc = OneHotEncoder(categories='auto')
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)


# test set
testdf_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)


#testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)

df_cat_data.head()

trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference

for col in difference:
    testdf_cat_data[col] = 0

print(df_cat_data.shape)    
print(testdf_cat_data.shape)

newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)

# test data
newdf_test=df_test.join(testdf_cat_data)
# newdf_test = df_test
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)

print(newdf.shape)
print(newdf_test.shape)

labeldf=newdf['label']
labeldf_test=newdf_test['label']


# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})


# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test

to_drop_DoS = [0,1]
to_drop_Probe = [0,2]


# Filter all rows with label value other than itself

DoS_df=newdf[newdf['label'].isin(to_drop_DoS)]
Probe_df=newdf[newdf['label'].isin(to_drop_Probe)]


#test
DoS_df_test=newdf_test[newdf_test['label'].isin(to_drop_DoS)]
Probe_df_test=newdf_test[newdf_test['label'].isin(to_drop_Probe)]



print('Train:')
print('Dimensions of DoS:' ,DoS_df.shape)
print('Dimensions of Probe:' ,Probe_df.shape)

print()
print('Test:')
print('Dimensions of DoS:' ,DoS_df_test.shape)
print('Dimensions of Probe:' ,Probe_df_test.shape)


# Split dataframes into X & Y


X_DoS = DoS_df.drop('label',1)
Y_DoS = DoS_df.label

X_Probe = Probe_df.drop('label',1)
Y_Probe = Probe_df.label


# test set
X_DoS_test = DoS_df_test.drop('label',1)
Y_DoS_test = DoS_df_test.label

X_Probe_test = Probe_df_test.drop('label',1)
Y_Probe_test = Probe_df_test.label


colNames=list(X_DoS)
colNames_test=list(newdf_test)

from sklearn import preprocessing

X_DoS.fillna(X_DoS.mean(), inplace=True)
X_Probe.fillna(X_Probe.mean(), inplace=True)
X_DoS_test.fillna(X_DoS_test.mean(), inplace=True)
X_Probe_test.fillna(X_Probe_test.mean(), inplace=True)

scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS) 

scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe)


scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test) 

scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test) 



#Random Forest (only DoS right now)

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10,n_jobs=2)

#DoS

np.nan_to_num(X_DoS_test)
np.nan_to_num(X_Probe_test)


clf_DoS = RandomForestClassifier(n_estimators=10,n_jobs=2)
clf_DoS.fit(X_DoS, Y_DoS.astype(int))
filename = 'finalized_model.sav'
pickle.dump(clf_DoS, open(filename, 'wb'))

clf_Probe = RandomForestClassifier(n_estimators=10,n_jobs=2)
clf_Probe.fit(X_Probe, Y_Probe.astype(int))

Y_DoS_pred = clf_DoS.predict(X_DoS_test)


Y_Probe_pred=clf_Probe.predict(X_Probe_test)

print('Total DoS attempts:: ', np.sum(Y_DoS_pred))
print('Total Probe attempts:: ', np.sum(Y_Probe_pred))

if (np.sum(Y_DoS_pred) > 10):
    print("Potential DoS attack detected.")

if (np.sum(Y_Probe_pred) > 10):
    print("Potential Probe attack detected.")
