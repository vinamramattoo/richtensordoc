## How to use this !!!!!
`To check out installation please check oldreadme.md .I am assuming that you have all libraries for tensorflow and protobuf set up`
#### 1. Get all images you want to put in the data set
#### 2. use LabelImg to create pascal Voc xml labelling data
#### 3. seperate the xml annotations and images in their corresponding folders
#### 4. run 
``` 
python xml_to_csv.py annotations/
``` 
#### 5.  this will generate the sample csv file
#### 6. run the split_labels.ipynb to split training and test data into seperate csv
#### 7. set $PYTHONPATH to get object detection model from tensorflow by going to models directory and setting
```
export PYTHONPATH=$PYTHONPATH:`pwd`/research:`pwd`/research/slim
```
#### 8. to generate tf records  alter the generate_tfrecord and change watchback to whatever model you are training then run 
```
python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record --image_dir=images  
```
##### same for test as well
#### 9. to run the training I used legacy code
```
python object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/training.config 
```

#### 10. This will generate the trained model to freeze it run
```
python object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path training/training.config --trained_checkpoint_prefix ../../../tmpjyp8mwta/model.ckpt-5000 --output_directory ./final_frozen_model.pb
```

#### 11. you can then run the detection on freezed model by altering and using 
```
python p2inference.py
```



##Explaination
`test_xml_to_csv.py` file scans through the annotations and generates appropriate csv

`split labels.ipynb` generates training and testing csv

`generate_tfrecord.py` convers the csv into binary format 'TFRecord' which is the suggested input type for tensorflow training we run this for both training and testing csv files

`object_detection/legacy/train.py` contains the legacy code for initiating training using tensorflow

While running the latest code which is in `object_detection/main_model.py` i found out it was not working with python3 'which is fixed in this codebase' and also I had issues with freezing the trained model it generated so I switched back to legacy codebase

under `training/training.config` file the configuration for `ssd_mobilenet_v1` has been defined to train our custom dataset from scratch there are a lot of configurations that can be tweaked in this file but lets go over the most important ones here

#### `num_steps:` defines how many steps should the training run for 
### set below for training data
```
train_input_reader: {
  tf_record_input_reader {
    input_path: "/home/vinamra/proj/richmont-tensorflow/data/train.record"
  }
  label_map_path: "/home/vinamra/proj/richmont-tensorflow/models/research/training/object-detection.pbtxt"
}
```
## set number of training images 
```
eval_config: {
  metrics_set: "coco_detection_metrics"
  num_examples: 542
}
```

## set below for eval data
```
eval_input_reader: {
  tf_record_input_reader {
    input_path: "/home/vinamra/proj/richmont-tensorflow/data/test.record"
  }
  label_map_path: "/home/vinamra/proj/richmont-tensorflow/models/research/training/object-detection.pbtxt"
  shuffle: false
  num_readers: 1
}
```

## if you have an existing checkpoint run finetuning by altering
```
fine_tune_checkpoint: "ENTER HERE"
```
