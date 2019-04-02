#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout


# In[2]:


#masukin data1
data1=pandas.read_csv('D:\\SKRIPSI\\data\\data_praproses\\jamu_herbs.csv', sep=',')
data1.values


# In[3]:


#masukin data2
data2=pandas.read_csv('D:\\SKRIPSI\\data\\data_praproses\\jamu_class.csv', sep=',')
data2.values


# In[4]:


data_1 = data1.drop('IDJamu',axis=1)
data_2 = data2.drop('Jamu ID',axis=1)


# In[5]:


data_1['Kelas']=data_2['Class of Diseases']


# In[6]:


X = data_1.drop('Kelas', axis=1).values
y = data_1['Kelas'].values


# In[7]:


from sklearn.model_selection import KFold


# In[8]:


from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[9]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[10]:


#untuk k=5
kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[11]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[12]:


#untuk k=6
kf = StratifiedKFold(n_splits=6, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[14]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[15]:


#untuk k=7
kf = StratifiedKFold(n_splits=7, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[16]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[17]:


#untuk k=8
kf = StratifiedKFold(n_splits=8, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[18]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[19]:


#untuk k=9
kf = StratifiedKFold(n_splits=9, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[20]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[21]:


#untuk k=10
kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
val_loss_cv = []
val_acc_cv = []
j = 0
for train_index, test_index in kf.split(X,y):
    j+=1
    print(f"Fold {j} :")
    print("")
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128,
                     kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))

    for i in range(0, 6):
        model.add(Dense(units=128, kernel_initializer='normal',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    model.add(Dense(units=19))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=24,epochs=30,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    val_loss_cv.append(score[0])
    val_acc_cv.append(score[1])


# In[22]:


print(f"val_loss_cv : {val_loss_cv}")
print(f"mean_val_loss_cv : {np.mean(val_loss_cv)}")
print(f"val_acc_cv : {val_acc_cv}")
print(f"mean_val_acc_cv : {np.mean(val_acc_cv)}")


# In[ ]:




