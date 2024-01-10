##### Importing libraries

import tensorflow as tf 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import load_model


##### Importing model

model = load_model('model.keras')


###### Loading data

data = 'mouth'

parse = keras.utils.image_dataset_from_directory(data)

###### Scaling data

parse = parse.map(lambda x,y : (x/255, y))
parsed_iterator =  parse.as_numpy_iterator()
batch = parsed_iterator.next()

print(batch)

fig, ax = plt.subplots(ncols=4, figsize=(12,12))
for index, img in enumerate(batch[0][:4]):
    ax[index].imshow(img)
    ax[index].title.set_text(batch[1][index])

print(ax[index].imshow(img))
print(ax[index].title.set_text(batch[1][index]))

print(len(parse))


train_size = int(len(parse)*0.7)
val_size = int(len(parse)*0.2)
test_size = int(len(parse)*0.1)

print(train_size, val_size, test_size)

print(train_size+val_size+test_size)

###### Blending data into train, val and test

train = parse.take(train_size)
val = parse.skip(train_size).take(val_size)
test = parse.skip(train_size+val_size).take(test_size)


print(train, val, test)


##### Evaluating model

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())
