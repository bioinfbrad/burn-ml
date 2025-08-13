import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
from tqdm import tqdm
#from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.image as mpimg

#os.chdir('/home/khaled/Desktop/firs_file/')



data_dir= sys.argv[1] #"/home/khaled/Downloads/Burn_mod/" #dir for the images
CATEGORIES=["Partial_thickness_burn","Normal_skin","Full_thickness_burn"] # creat a list for labiling  creat a list for labiling
IMG_SIZE=50

training_data = []
rotate_img=[]
part_ev=[]
part=[]
normal_ev=[]
normal=[]
full_ev=[]
full=[]
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
         results[i, label] = 1.
    return results

def create_training_data():
    imeges=[]

    for num in range (0,360,15):
        for category in CATEGORIES:  # do Normal skin, Partial thickness burn or full thickness burn
            if category == "Partial_thickness_burn":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  # get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image in partial thickniss burn
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        part.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))

            elif category == "Normal_skin":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  #get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image in normal skin
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        normal.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))

            elif category == "Full_thickness_burn":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  #get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image per Full thickniss burn
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        full.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))



create_training_data() # run the created function to extact the data





# shuffle the data

import random
random.shuffle(part)
random.shuffle(normal)
random.shuffle(full)


# extract the evaluation data from the training data
part_ev=part[0:100]

part=part[100:]

normal_ev=normal[0:100]
normal=normal[100:]

full_ev=full[0:100]
full=full[100:]


training_data=part+normal+full
random.shuffle(training_data)

test_data=part_ev+normal_ev+full_ev
random.shuffle(test_data )




# extract the x (training data) and y (labels) to introduce them to the model
#x for feture y for label

x=[]
y=[]


for features, label in training_data:#the list has tow part first is the image the second is the label
    x.append(features)
    y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
y = to_one_hot(y)
x=np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3) #-1 is the numper of features, 1 is the gray scale  #this is were the tensor created

x = x/255.0

part_x=[]
part_y=[]
for features, label in part_ev:#the list has tow part first is the image the second is the label
    part_x.append(features)
    part_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
part_hot_y = to_one_hot(part_y)
part_x=np.array(part_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)



part_x=part_x/255.0

nor_x=[]
nor_y=[]
for features, label in normal_ev:#the list has tow part first is the image the second is the label
    nor_x.append(features)
    nor_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
nor_hot_y = to_one_hot(nor_y)
nor_x=np.array(nor_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)
nor_x=nor_x/255.0

full_x=[]
full_y=[]
for features, label in full_ev:#the list has tow part first is the image the second is the label
    full_x.append(features)
    full_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
full_hot_y = to_one_hot(full_y)
full_x=np.array(full_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)
full_x=full_x/255.0


x_test=[]
y_test=[]
for features, label in test_data:#the list has tow part first is the image the second is the label
    x_test.append(features)
    y_test.append(label)
#because we cant feed a list into the CNN we have to convert it into array
y_test = to_one_hot(y_test)
x_test=np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,3)
x_test=x_test/255.0

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
from tqdm import tqdm
#from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

opt = tf.keras.optimizers.Adam(learning_rate=0.001)



dense_layers = [2]
layer_sizes = [128]
conv_layers = [4]
batch_sizes=[64]


k = 4
num_val_samples = len(x) // k
num_epochs = 15
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = x[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [x[:i * num_val_samples],
    x[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
    [y[:i * num_val_samples],
    y[(i + 1) * num_val_samples:]],
    axis=0)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))


    model.add(Conv2D(64, (3, 3)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    #model.add(Dropout(0.1))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=64, verbose=0) #, callbacks=[tensorboard])
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)



print("full_thickness_evaluation")
model.evaluate(full_x,full_hot_y)

print("normal_skin_evaluation")
model.evaluate(nor_x,nor_hot_y)

print("full_thickness_evaluation")
model.evaluate(part_x,part_hot_y)


# partial evaluation
print("partial_thicknes_evaluation")
result = model.predict_proba(part_x)

model.evaluate(part_x,part_hot_y)
for i in range(len(part_y)):



    if part_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        font_color = 'red' if part_y[i] != np.argmax(result[i]) else 'black'
        plt.text(x=-20, y=-10, s=CATEGORIES[int(part_y[i])], fontsize=18, color="black")
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(part_y[i])])
        plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(part_x[i],cmap='gray')
        fig.savefig('part_miss_predicted'+str(i)+ '.png')
        plt.show()

    elif part_y[i] == np.argmax(result[i]):
        fig= plt.figure()
        plt.title("the correct catigory is "+CATEGORIES[int(part_y[i])])
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.text(x=-20, y=-10, s=CATEGORIES[int(part_y[i])], fontsize=18, color="black")
        plt.imshow(part_x[i],cmap='gray')
        plt.savefig('part_correct_predicted'+str(i)+ '.png')
        plt.show()


# normal skin evaluation
print("normal_skin_evaluation")
result = model.predict_proba(nor_x)

model.evaluate(nor_x,nor_hot_y)

for i in range(len(nor_y)):
    if nor_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        font_color = 'red' if nor_y[i] != np.argmax(result[i]) else 'black'
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(nor_y[i])])
        plt.text(x=-20, y=-10, s=CATEGORIES[int(nor_y[i])], fontsize=18, color="black")
        plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(nor_x[i],cmap='gray')
        plt.savefig('nor_miss_predicted'+str(i)+ '.png')
        plt.show()

    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(nor_y[i])])
        plt.text(x=-20, y=-10, s=CATEGORIES[int(nor_y[i])], fontsize=18, color="black")
        plt.imshow(nor_x[i],cmap='gray')
        plt.savefig('nor_correct_predicted'+str(i)+ '.png')
        plt.show()


# full thickness evaluate
print("full_thicknes_evaluation")
result = model.predict_proba(full_x)


model.evaluate(full_x,full_hot_y)

for i in range(len(full_y)):
    if full_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(full_y[i])])
        font_color = 'red' if full_y[i] != np.argmax(result[i]) else 'black'
        plt.text(x=-20, y=-10, s=CATEGORIES[int(full_y[i])], fontsize=18, color="black")
        plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(full_x[i],cmap='gray')
        plt.savefig('full_miss_predicted'+str(i)+ '.png')
        plt.show()


    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(full_y[i])])
        plt.text(x=-20, y=-10, s=CATEGORIES[int(full_y[i])], fontsize=18, color="black")
        plt.imshow(full_x[i],cmap='gray')
        plt.savefig('full_correct_predicted'+str(i)+ '.png')
        plt.show()




print(all_scores)
print(np.mean(all_scores))



#add figures

print("full_thickness_evaluation")
model.evaluate(full_x,full_hot_y)

print("normal_skin_evaluation")
model.evaluate(nor_x,nor_hot_y)

print("full_thickness_evaluation")
model.evaluate(part_x,part_hot_y)






all_mae_histories=[]

mae_history = history.history['loss']
all_mae_histories.append(mae_history)



average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.savefig('val_loss.png')
plt.show()




all_mae_histories=[]

mae_history = history.history['accuracy']
all_mae_histories.append(mae_history)



average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
ns_probs = [0 for _ in range(len(full_hot_y))]



plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.savefig('accuracy.png')
plt.show()




#Results ROC CURVE

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc



y_score = model.predict(x_test)
# Plot linewidth.
lw = 2
n_classes=3
classes=["Partial_thickness_burn","Normal_skin","Full_thickness_burn" ]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig= plt.figure(1)
fig.patch.set_facecolor('xkcd:white')
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of  {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()


# Zoom in view of the upper left corner.
fig= plt.figure(2)
fig.patch.set_facecolor('xkcd:white')
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve (upper left corner)')
plt.legend(loc="lower right")
plt.savefig('Zoomed.png')
plt.show()



# Confusion matrix

y_pred = model.predict(x_test)
y_pred_1=np.argmax(y_pred, axis=1)

#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred_1)
print('Confusion Matrix\n')
print(confusion)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = confusion
df_cm = pd.DataFrame(array, index = [i for i in ["Partial_thickness_burn","Normal_skin","Full_thickness_burn"]],
                  columns = [i for i in ["Partial_thickness_burn","Normal_skin","Full_thickness_burn"]])
fig= plt.figure(figsize = (10,7))
fig.patch.set_facecolor('xkcd:white')
sn.heatmap(df_cm, annot=True)
plt.savefig('Confusion_matrix.png')
