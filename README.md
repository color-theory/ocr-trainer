# ocr-trainer
Train an OCR model with TensorFlow.
- Note: This project will likely not function properly on windows 11 unless you are in the windows subsystem for linux.

## Usage
```
ocr_trainer> python main.py
```
This will save your model as `ocr_model` in your ocr_trainer directory. 
It will also save a `label_mapping.json` file for you to use if you wish to associate the predicted indexes with their labels elsewhere.

## Convert to web model for use with tensorflow.js
```
ocr_trainer> tensorflowjs_converter --input_format=tf_saved_model ./ocr_model ./web_model
```