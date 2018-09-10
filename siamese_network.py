# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:11:42 2018

@author: LDUPUIS
"""


import pickle
import os
import boto3
import botocore
import numpy as np

## --- Test Keras : CNN
    
#import



from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Lambda, Concatenate
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.utils import generic_utils
from keras.models  import  Sequential
from keras.layers.convolutional  import Conv1D ,  MaxPooling1D
from keras.regularizers import L1L2
from keras import backend as K
##keep list of media with boto3

s3 = boto3.client("s3")
list_media = []
compteur =1
objets = s3.list_objects(Bucket = "midroll-ftven", Prefix = "index_features")
liste_objets = [dico["Key"] for dico in objets["Contents"]]
list_media_tmp = [elem.split("index_features/")[1] for elem in liste_objets if elem.split("index_features/")[1]!=""]
list_media += list_media_tmp
print("iter %s : nb de media total %s" % (compteur,len(list_media)))
while objets["IsTruncated"]: 
    objets = s3.list_objects(Bucket = "midroll-ftven",Prefix = "index_features", Marker = objets["Contents"][int(objets["MaxKeys"])-1]["Key"])
    compteur += 1
    liste_objets = [dico["Key"] for dico in objets["Contents"]]
    list_media_tmp = [elem for elem in liste_objets]
    list_media += list_media_tmp
    print("iter %s : nb de media = %s" % (compteur,len(list_media)))

print(list_media)

data_train=[]
list_scene_train=[]
list_shot_train=[]

data_test=[]
list_scene_test=[]
list_shot_test=[]

BUCKET_NAME = 'midroll-ftven' # replace with your bucket name
count=1

nb_train=int(len(list_media)*0.7)

for media in list_media:
    print("media " + str(media))

    KEY_features = 'index_features/'+media # replace with your object key
    KEY_shot = 'index_shot/shot_'+media # replace with your object key
    KEY_scene = 'index_scene/scene_'+media # replace with your object key
    
    KEY_features_2 = media # replace with your object key
    KEY_shot_2 = 'shot_'+media # replace with your object key
    KEY_scene_2 = 'scene_'+media # replace with your object key
    
    s3 = boto3.resource('s3')
    
    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY_features, KEY_features_2)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    
    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY_shot, KEY_shot_2)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    
    
    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY_scene, KEY_scene_2)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
            

#    #download features, shots and scenes for each media
#    #print("aws s3 cp s3://midroll-ftven/index_shot/shot_%s shot_%s" % (media, media))
#    os.system("aws s3 cp s3://midroll-ftven/index_shot/shot_%s shot_%s" % (media, media))
#    os.system("mkdir dir_"+str(media))
#    #print("aws s3 cp s3://midroll-ftven/index_scene/scene_%s scene_%s" % (media, media))
#    os.system("aws s3 cp s3://midroll-ftven/index_scene/scene_%s scene_%s" % (media, media))
#    #print("aws s3 cp s3://midroll-ftven/index_features/%s %s" % (media, media))
#    os.system("aws s3 cp s3://midroll-ftven/index_features/%s %s" % (media, media))
#    print('download ok')
#    #open 3 dictionnary
    
    print('open ' +str(media))
    with open(media,"rb") as file :
        features = pickle.load(file) 
#    f = open(media,"rb")
#    features = pickle.load(f)
#    f.close()
        

    print('open scene_' +str(media))
    with open("scene_"+media,"rb") as file :
        scene = pickle.load(file) 
#    f = open("scene_"+media,"rb")
#    scene = pickle.load(f)
#    f.close()
    
    print('open shot_' +str(media))
    with open("shot_"+media,"rb") as file :
        shot = pickle.load(file) 
#    f = open("shot_"+media,"rb")
#    shot = pickle.load(f)
#    f.close()
    
    #add on each list
    if count<=nb_train:
        data_train.append(features)
        list_scene_train.append(scene)
        list_shot_train.append(shot)
    else:
        data_test.append(features)
        list_scene_test.append(scene)
        list_shot_test.append(shot)
    
    del features, scene, shot
#    print("remove")
#    print(os.listdir())
#    print("rm shot_%s" % (media))
#    os.system("rm shot_%s" % (media))
#    print("rm scene_%s" % (media))
#    os.system("rm scene_%s" % (media))
#    print("rm %s" % (media))
#    os.system("rm %s" % (media))
    
    count+=1
    print(count)
    os.remove(media)
    os.remove("shot_"+media)
    os.remove("scene_"+media)


#filename_1 = 'data.p'
#joblib.dump(data, filename_1)
#print("save data")
#filename_2 = 'list_scene.p'
#joblib.dump(list_scene, filename_2)  
#print("save list_scene")
#filename_3 = 'list_shot.p'
#joblib.dump(list_shot, filename_3)  
#print("save list_shot")



# 2 shots in the same scene or not
def shots_same_scene(scene_list, nb_scene, tmp_shot_right):
    if nb_scene<len(scene_list):
        if scene_list[nb_scene]>tmp_shot_right:
            return True
        else:
            return False
    else:
        return True
    
#create the data input : pair of data (shot_right and shot_left)
def create_pair(data, list_scene, list_shot):
    pairs = []
    labels = []
    for media in range(len(data)):
        shot=1
        nb_scene_current=0
        while (shot<len(data[media])):
            pairs+=[[data[media][shot-1],data[media][shot]]]
            temp_shot_right=list_shot[media][shot][1]
            if shots_same_scene(list_scene[media], nb_scene_current, temp_shot_right): #same scene
                labels.append(0)
            else:
                nb_scene_current+=1 #other scene
                labels.append(1)
            shot+=1
    return np.array(pairs), np.array(labels)

#create the data input : pair of data (shot_right and shot_left)
def create_pair_key(data, list_scene, list_shot, key_dic):
    pairs = []
    labels = []
    for media in range(len(data)):
        shot=1
        nb_scene_current=0
        while (shot<len(data[media])):
            pairs+=[[data[media][shot-1][key_dic],data[media][shot][key_dic]]]
            temp_shot_right=list_shot[media][shot][1]
            if shots_same_scene(list_scene[media], nb_scene_current, temp_shot_right): #same scene
                labels.append(0)
            else:
                nb_scene_current+=1 #other scene
                labels.append(1)
            shot+=1
    return np.array(pairs), np.array(labels)



def convert(res):   
    newarray = np.dstack(res) # 1 x 2 x nb_shots x 3 ==> nb_shots x 2 x 1 x 3
    newarray = np.rollaxis(newarray, 1) # on réordonne pour avoir les données sous la forme Nx2x1x2048
    newarray = np.rollaxis(newarray, 2) # on réordonne pour avoir les données sous la forme Nx2x1x2048
    return newarray


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 3.14
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def create_model():
    """Creates the neural network.
    Returns:
        neural network model, optimizer
    """
    dense_regularizer = L1L2(l2=0.0005)

    audio_input = Input(shape=(1, 128, 301),dtype="float64")
    audio= Conv1D(32,kernel_size=(128,8),padding="same",activation="relu",name="Conv1D_32/128/8_relu")(audio_input)
    audio= MaxPooling1D(pool_size =(1,4),name="MaxPooling1D_1/4_1")(audio)
    
    audio=Conv1D(32,kernel_size=(1,8),padding="same",activation="relu",name="Conv1D_32/1/8_relu")(audio)
    audio=MaxPooling1D(pool_size =(1,2),name="MaxPooling1D_1/2_2")(audio)
       
    audio=Conv1D(32,kernel_size=(1,4),padding="same",activation="relu",name="Conv1D_32/1/4_relu")(audio)
    audio=MaxPooling1D(pool_size =(1,2),name="MaxPooling1D_1/2_3")(audio)
    
    audio=Dense(1024, activation="relu", name="Dense_1024",kernel_regularizer=dense_regularizer)(audio)
    audio=Dense(512,activation=" relu",name=" Dense_512",kernel_regularizer=dense_regularizer)(audio)
    audio_output=audio
    
    cnn_audio_model=Model(audio_input, audio_output)

    
    input_audio=Input(shape=(128,301))
    input_image=Input(shape=(1,2048))
    input_time=Input(shape=(1))
    
    list_output_audio=K.map_fn(lambda feature: cnn_audio_model, input_audio)
    
    #element wise
    output_audio=np.maximum.reduce(list_output_audio)
    
    
    merged=Concatenate([input_image, output_audio, input_time])
    
    
    X_input = Input(merged)
    
    X=Dense(100, activation="relu", name="Dense_100_1",kernel_regularizer=dense_regularizer)(X_input)
    X=Dense(100,activation=" relu",name=" Dense_100_2",kernel_regularizer=dense_regularizer)(X)
    X_output=X

    return Model([input_audio, input_image, input_time], X_output)


epochs=20


## create data train and data validation

## create data train and data test
    #train
print("create data train et data validation")
x_train, y_train=create_pair(data_train, list_scene_train, list_shot_train)
x_train=convert(x_train)
    #validation
x_test, y_test=create_pair(data_test, list_scene_test, list_shot_test)
x_test=convert(x_test)


#train
x_image_train,y_image_train=create_pair_key(data_train,list_scene_train,list_shot_train, "image")
x_audio_train,y_audio_train=create_pair_key(data_train,list_scene_train,list_shot_train, "audio")
X_audio_train=convert(x_audio_train)
x_time_train,y_time_train=create_pair_key(data_train,list_scene_train,list_shot_train, "time")
X_time_train=convert(x_time_train)

#validation
x_image_test,y_image_test=create_pair_key(data_test,list_scene_test,list_shot_test, "image")
x_audio_test,y_audio_test=create_pair_key(data_train,list_scene_test,list_shot_test, "audio")
X_audio_test=convert(x_audio_test)
x_time_test,y_time_test=create_pair_key(data_test,list_scene_test,list_shot_test, "time")
X_time_test=convert(x_time_test)



input_shape_audio=(128,301)
input_shape_image=(1,2048)
input_shape_time=(1,1)

# network definition
print("create siamese network")
siamese_network = create_model()

input_audio_left = Input(shape=input_shape_audio)
input_image_left = Input(shape=input_shape_image)
input_time_left = Input(shape=input_shape_time)

input_audio_right = Input(shape=input_shape_audio)
input_image_right = Input(shape=input_shape_image)
input_time_right = Input(shape=input_shape_time)



# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
print("processed")
processed_left = siamese_network([input_audio_left, input_image_left,input_time_left])
processed_right = siamese_network([input_audio_right, input_image_right, input_time_right])

print("distance")
distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_left, processed_right])
# Siamese Network
print("model")
model = Model([[input_audio_left, input_image_left,input_time_left], [input_audio_right, input_image_right, input_time_right]], distance)

# optimizer

adam=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay= 0.96, amsgrad=False)

print("compile")
model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
print("fit")
model.fit([[x_audio_train[:, 0], x_image_train[:, 0], x_time_train[:, 0]],[x_audio_train[:, 1], x_image_train[:, 1], x_time_train[:, 1]]], y_train,batch_size=128,epochs=epochs,
          validation_data=([[x_audio_test[:, 0], x_image_test[:, 0], x_time_test[:, 0]],[x_audio_test[:, 1], x_image_test[:, 1], x_time_test[:, 1]]], y_test))

# compute final accuracy on training and test sets
print("compute accuracy")
y_pred = model.predict([[x_audio_train[:, 0], x_image_train[:, 0], x_time_train[:, 0]],[x_audio_train[:, 1], x_image_train[:, 1], x_time_train[:, 1]]])
tr_acc = compute_accuracy(y_train, y_pred)
y_pred = model.predict([[x_audio_test[:, 0], x_image_test[:, 0], x_time_test[:, 0]],[x_audio_test[:, 1], x_image_test[:, 1], x_time_test[:, 1]]])
te_acc = compute_accuracy(y_test, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))



#filename_1 = 'x_train.p'
#joblib.dump(x_train, filename_1)
#print("save x_train")
#filename_2 = 'y.p'
#joblib.dump(y, filename_2)  
#print("save y")
#
#print("stop save ")
