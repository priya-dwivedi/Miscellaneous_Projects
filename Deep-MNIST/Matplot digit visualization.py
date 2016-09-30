
# coding: utf-8

# In[1]:

data_file = open("C:\Users\s6324900\Desktop\Deep learning\my_neural_net\mnist_train_100.csv", 'r') 
data_list = data_file.readlines() 
data_file.close()


# In[2]:

len(data_list)


# In[3]:

data_list[0]


# In[5]:

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib inline')


# In[7]:

all_values= data_list[0].split(',')
image_array= np.asfarray(all_values[1:]).reshape((28,28))


# In[10]:

plt.imshow(image_array, cmap='Greys', interpolation='None')


# In[11]:

all_values= data_list[1].split(',')
image_array= np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')


# In[ ]:



