#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_csv("penguins_size.csv")


# In[7]:


df.info()
df.shape


# In[12]:


df.isnull().sum()


# In[15]:


for i in df.columns[2:6]:
    df[i].fillna(value=df[i].mean(), inplace=True)


# In[19]:


df["sex"].fillna(value=df["sex"].mode()[0], inplace=True)
df


# In[21]:


df.describe(include="all").T


# In[24]:


df.info()
df.isnull().sum()


# In[28]:


plt.figure()
sns.pairplot(data=df,hue="species")
plt.show()


# In[29]:


plt.figure()
sns.pairplot(data=df, hue="island")
plt.show()


# In[36]:


for i in df.columns[2:6]:
    plt.figure()
    skewness= round(df[i].skew(),3)
    sns.distplot(df[i])
    plt.title(f"distribution of {i} | skewness= {skewness}")
    plt.show()


# In[39]:


for i in df.columns[2:6]:
    plt.figure()
    Q3= df[i].quantile(0.75)
    Q1= df[i].quantile(0.25)
    IQR= round(Q3-Q1, 3)
    sns.boxplot(df[i])
    plt.title(f"distribution of {i} | IQR= {IQR}")
    plt.show()


# In[48]:


df["sex"]=df["sex"].replace({".":"Unknown"})
df["sex"].unique()


# In[49]:


for i in df.select_dtypes("object").columns:
    plt.figure()
    sns.countplot(df[i])
    plt.title(f"countplot of {i}")
    plt.show()


# In[73]:


from scipy.stats import f_oneway

for i in df.columns[2:6]:
    group1= df[df["island"]=="Torgersen"][i]
    group2= df[df["island"]=="Biscoe"][i]
    group3= df[df["island"]=="Dream"][i]
    result=f_oneway(group1,group2,group3)
    print(f"{i}, result: {result.statistic: .3f}, p-value: {result.pvalue: .3f}")


# In[78]:


from scipy.stats import f_oneway

for i in df.columns[2:6]:
    group1= df[df["species"]=="Adelie"][i]
    group2= df[df["species"]=="Chinstrap"][i]
    group3= df[df["species"]=="Gentoo"][i]
    result=f_oneway(group1,group2,group3)
    print(f"{i}, result: {result.statistic: .3f}, p-value: {result.pvalue: .3f}")


# In[79]:


from scipy.stats import f_oneway

for i in df.columns[2:6]:
    group1= df[df["sex"]=="MALE"][i]
    group2= df[df["sex"]=="FEMALE"][i]
    result= f_oneway(group1, group2, group3)
    print(f"{i}, result: {result.statistic: .3f}, p-value: {result.pvalue: .3f}")


# In[93]:


sns.heatmap(df[df.columns[2:6]].corr(),annot=True)
plt.tight_layout()
plt.show()


# In[105]:


plt.figure()
plt.pie(df["species"].value_counts(),labels=df["species"].unique(), autopct="%1.1f%%")
plt.show()


# In[106]:


df["island"].value_counts()


# In[117]:


plt.figure()
sns.set_style("darkgrid")
plt.pie(df["island"].value_counts(),labels= df["island"].unique(), autopct="%1.1f%%", explode=[0.05,0.05,0.05])
plt.show()


# In[ ]:




