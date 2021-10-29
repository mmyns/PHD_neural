import Load_data
import argparse
import json
import pickle
import Model1
import tensorflow as tf
import datetime

log_dir = "/esat/biomeddata/guests/r0376890/tensor_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("config", help="gets config file")

args = parser.parse_args()
with open("/esat/biomeddata/guests/r0376890/condor_logs/" + args.config) as json_data_file:
    data = json.load(json_data_file)

print(data)

filename = data["filename"]
selected_labels = data["labels"]
batch_size = data["batch_size"]
frames = data["frames"]
final_layer = len(selected_labels)
label_smoothing = data['smoothing']
split_number = data["split_number"]
dropout_rate = data["dropout"]
lr = data["learning-rate"]

if data["loss"] == "Categorical":
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=label_smoothing)
    final_activation = "linear"
    final_size = 4
elif data["loss"] == "Binary":
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=label_smoothing)
    final_activation = "sigmoid"
    final_size = 1

if "resize" in data:
    input_shape = (data["resize"][0],data["resize"][1],len(frames))
else:
    input_shape = (1012, 1356, len(frames))


f = open('/esat/biomeddata/guests/r0376890/Validation_test_split.pkl', 'rb')
splits = pickle.load(f)
f.close()

dataset_list = Load_data.getfiles(frames)
train_dataset,validation_dataset = Load_data.split_and_label_data(dataset_list, batch_size, selected_labels,splits)

model = Model1.create_model(input_shape, dropout_rate, final_activation)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=loss_func,
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'/esat/biomeddata/guests/r0376890/tmp/{filename}-bestmodel.h5',
    verbose = 1,
    monitor='val_auc',
    save_weights_only= True,
    mode='max',
    save_best_only=True)

csv_callback = tf.keras.callbacks.CSVLogger(f'/esat/biomeddata/guests/r0376890/tmp/csv/{filename}.csv', separator=",", append=False)


history = model.fit(
    train_dataset,
    epochs=150,
    verbose=2,
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback,csv_callback]
)
