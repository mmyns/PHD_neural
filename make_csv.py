import pickle
import pandas as pd
import os
import glob

frames = [1,10,19,29,34]
selected_labels = ["All","ABMRh","Cellular"]

f = open('/esat/biomeddata/guests/r0376890/Validation_test_split.pkl', 'rb')
splits = pickle.load(f)
f.close()

root_folder = r'/esat/biomeddata/guests/r0376890/opal_data'
label_folder = r'/esat/biomeddata/guests/r0376890/opal_data/labels'
root_color_folder = r'/esat/biomeddata/mmaeyens/OPAL/Color_images'
label_csv= pd.read_csv(os.path.join(label_folder,'processed_labels.csv'))


output_dict = {"file":[],"split":[]}
for frame in frames:
    output_dict[str(frame)] = []
for label in selected_labels:
    output_dict[label] = []
for i,split in enumerate(splits):
    for biopsy in split:

        glob_output =  glob.glob(str(root_folder+ os.sep + f'{biopsy}/*/'))
        sampled_list = glob_output
        for sample in sampled_list:
            output_dict["split"].append(i)
            output_dict["file"].append(sample)
            for label in selected_labels:
                label_value = label_csv[label_csv['Biopsy number'] == biopsy][label].iloc[0]
                output_dict[label].append(label_value)
            for frame in frames:
                output_dict[str(frame)].append(os.path.join(sample + f'{frame}',os.listdir(sample + f'{frame}')[0]))

df = pd.DataFrame.from_dict(output_dict)
df.to_csv("/esat/biomeddata/mmaeyens/logs/filesandlabels.csv")

