import pandas as pd
import os
import tensorflow as tf
import numpy as np

root_folder = r'/esat/biomeddata/guests/r0376890/opal_data'
label_folder = r'/esat/biomeddata/guests/r0376890/opal_data/labels'
label_csv= pd.read_csv(os.path.join(label_folder,'processed_labels.csv'))


def getfiles(frames):
    dataset_list = []
    for frame in frames:
        temp_dataset = tf.data.Dataset.list_files(str(root_folder + r'/' + f'*/*/{frame}/*'), shuffle=False)
        dataset_list.append(temp_dataset)
    return dataset_list

def get_label(biopsy,selected_label,loss):
  label_cell = label_csv[label_csv['Biopsy number'] == biopsy.numpy().decode("utf-8") ][selected_label.numpy().decode("utf-8")].iloc[0]
  if loss == "Categorical":
    label_cell  = tf.keras.utils.to_categorical(label_cell,num_classes=4)
  else:
    label_cell = label_cell !=0
  return label_cell

def checkstring(x,list):
    return x.numpy().decode("utf-8") in list

def process_path(file_paths,label_cell):
  inputs = []
  for file in file_paths:
    files = tf.io.read_file(file)
    img = tf.io.decode_png(files, channels=0, dtype=tf.dtypes.uint16)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=False)
    inputs.append(img)

  images = tf.concat(inputs,axis=2)
  return images, label_cell

def labeldata(data,sublist,selected_labels):
    dataset_list = []
    for ds in data:
        temp = ds.filter(lambda x: tf.py_function(checkstring,[tf.strings.split(x, os.sep)[-4],sublist],tf.bool))
        dataset_list.append(temp)
    finalds = tf.data.Dataset.zip(tuple(dataset_list))
    mega_list = []
    for tds in finalds:
        biopsy = tf.strings.split(tds[0], os.sep)[-4]
        label_list = []
        for selected_label in selected_labels:
            label_cell = tf.py_function(get_label, [biopsy, selected_label], [tf.int8])
            label_list.append(label_cell)
        label_cell = tf.convert_to_tensor(label_list, dtype=tf.int8)
        if data["loss"] == "Binary":
            label_cell = tf.reshape(label_cell, [])
        label_cell = tf.squeeze(label_cell)
        mega_list.append(label_cell)
    label_ds = tf.data.Dataset.from_tensor_slices(mega_list)
    finalds = tf.data.Dataset.zip((finalds, label_ds))
    return finalds

def split_and_label_data(data,batch_size, selected_labels, splits = None, validate_split = None):
    if splits == None:
        all_files = os.listdir(root_folder)
        all_files.remove('labels')
        size_all_files = len(all_files)
        np.random.shuffle(all_files)
    else:
        if validate_split == None:
            raise Exception("No validate split assigned")

        train = []
        validate = splits[validate_split]
        for j in range(0, 5):
            if j != validate_split:
                train.extend(splits[j])
    trainds = labeldata(data,train,selected_labels)
    validateds = labeldata(data, train, selected_labels)


    label_train = trainds.shuffle(2000, reshuffle_each_iteration=False)
    label_train = label_train.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    label_validate = validateds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = label_train.batch(batch_size, drop_remainder=True).cache()
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE).shuffle(200, reshuffle_each_iteration=True)
    validation_dataset = label_validate.batch(batch_size, drop_remainder=True).cache()
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset,validation_dataset