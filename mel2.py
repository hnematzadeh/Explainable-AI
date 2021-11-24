import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import display # Allows the use of display() for DataFrames
from time import time
import matplotlib.pyplot as plt
import seaborn as sns # Plotting library
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array,array_to_img
from keras.utils import np_utils
from sklearn.datasets import load_files   
from tqdm import tqdm
from collections import Counter
from sklearn.utils import resample, shuffle
from tensorflow.keras.applications.vgg16 import VGG16



print(os.listdir("/home/hossein/skin-lesions/train/"))


data_train_path = '/home/hossein/skin-lesions/train/'
# data_train_path = '/home/hossein/skin-lesions/Train2/'
data_valid_path = '/home/hossein/skin-lesions/valid/'
data_test_path =  '/home/hossein/skin-lesions/test/'


# Utest_tensors = test_tensors
# Utest_targets = test_targets

#================== get the class indices


# Class name to the index
#class_2_indices = train_generator.class_indices
class_2_indices = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratoses': 2}
print("Class to index:", class_2_indices)

# Reverse dict with the class index to the class name
indices_2_class = {v: k for k, v in class_2_indices.items()}
print("Index to class:", indices_2_class)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.utils import shuffle
import h5py
import tensorflow.keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


##############################PRE TRAINED MODEL RESNET##################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from keras_tqdm import TQDMNotebookCallback

base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dropout(0.2)(x)    #new
# x=Conv2D(filters=128, kernel_size=3, activation='relu',strides=1)(x)   #new [error]
# x = Dense(1024, activation='relu')(x)   #new

# x=keras.layers.MaxPool2D(pool_size=2,strides=2)(x)   #new [error]
# x = Dropout(0.5)(x)   #new

# x = MaxAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='elu')(x)
# x = Dense(1024, activation=tf.nn.relu)(x)     #new 
# and a logistic layer
x = Dropout(0.95)(x)
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = True
    
# for i in range(len(model.layers)-1):
#     model.layers[i].trainable=False     
    
from tensorflow.keras.optimizers import Adam

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
             metrics=['accuracy'])






# load the weights that yielded the best validation accuracy
model.load_weights('aug_model.weights.best.hdf5')
score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)
yhat=model.predict(X1_test)
print('\n', 'Test accuracy:', score[1])
###########################LOAD SAMPLE IMAGE###################################

## sample image in test with its real label
plt.imshow(X1_test[457])
# plt.imshow((X_test[14] * 255).astype(np.uint8))
image_idx = np.argmax(y1_test[457])
plt.title(indices_2_class[image_idx])
plt.axis('off')


#############################LIME#############################

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic, quickshift, watershed, felzenszwalb    
import random

explainer=lime_image.LimeImageExplainer()
explanation=explainer.explain_instance(X1_test[457],model.predict, num_samples=10000,segmentation_fn=quickshift )
explanation=explainer.explain_instance(X1_test[457],classifier_fn=model.predict, hide_color=0, num_samples=5000,  segmentation_fn=slic)
explanation=explainer.explain_instance(X1_test[457],model.predict, hide_color=0, num_samples=5000, segmentation_fn=felzenszwalb)
# explanation=explainer.explain_instance(X1_test[18],model.predict, hide_color=0, num_samples=50, segmentation_fn=watershed)


image, mask=explanation.get_image_and_mask(model.predict(X1_test[457].reshape((1,224,224,3))).argmax(axis=1)[0], positive_only=True, num_features=5,hide_rest=False)      


plt.imshow(mark_boundaries(array_to_img(image), mask))
plt.axis('off')


##############################################################


###########################SHAP###############################

import shap
import time


######################Deep Explainer for one observation#######
## DeepExplainer only works with SHAP version 0.31.0
class_names={0:'melanoma',1:'nevus',2:'seborrheic_keratoses'}
start=time.time()
background = X1_train[np.random.choice(X1_train.shape[0],30, replace=False)]
e = shap.DeepExplainer(model,background)
# add one dimension to left to make diemsions of background and sample same
sample_to_explain = np.expand_dims(X1_test[457], axis=0)
shap_values,indexes = e.shap_values(sample_to_explain,ranked_outputs=3)
index_names=np.vectorize(lambda x: class_names[x])(indexes)
shap.image_plot(shap_values,sample_to_explain,index_names)
end=time.time()

print(end-start)

############Gradient Explainer for one observation#############
## GradientExplainer only works with SHAP version 0.36.0  and 0.31.0 and...

class_names={0:'melanoma',1:'nevus',2:'seborrheic_keratoses'}
start=time.time()
background = X1_train[np.random.choice(X1_train.shape[0],4194, replace=False)]
e = shap.GradientExplainer(model,background,local_smoothing=0)
sample_to_explain = np.expand_dims(X1_test[457], axis=0)
shap_values,indexes = e.shap_values(sample_to_explain, ranked_outputs=3)
index_names=np.vectorize(lambda x: class_names[x])(indexes)
shap.image_plot(shap_values,sample_to_explain,index_names)
end=time.time()

print(end-start)










start=time.time()

explainer=lime_image.LimeImageExplainer()
# explanation=explainer.explain_instance(X1_test[457],model.predict, num_samples=4194,segmentation_fn=quickshift )
# explanation=explainer.explain_instance(X1_test[457],classifier_fn=model.predict, hide_color=0, num_samples=10000,  segmentation_fn=slic)
explanation=explainer.explain_instance(X1_test[457],model.predict, hide_color=0, num_samples=10000, segmentation_fn=felzenszwalb)


image, mask=explanation.get_image_and_mask(model.predict(X1_test[457].reshape((1,224,224,3))).argmax(axis=1)[0], positive_only=True, num_features=5,hide_rest=False)      
end=time.time()

print(end-start)

plt.imshow(mark_boundaries(array_to_img(image), mask))
plt.axis('off')
















