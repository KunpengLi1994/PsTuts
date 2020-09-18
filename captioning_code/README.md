# Tutorial Clip Captioning

We recommended the following dependencies

## requirements

- cuda
- pytorch 0.4.0
- python3
- ffmpeg

### python packages

- tqdm
- pillow
- pretrainedmodels
- nltk

## Data
All the data needed for reproducing the experiments, including video features and vocabularies, can be downloaded from google drive: https://drive.google.com/file/d/1GT3p-REVkHmPBXlgxEGFjEoUKuVcUhjZ/view?usp=sharing

We refer to the path of extracted files for `caption_data.zip` as `$DATA_PATH`, which can be put in the under the code folder. 

## Options
All default options are defined in opt.py or corresponding code file, change them for your like.



## Training new models

```bash

python train.py --gpu 0 --epochs 301 --batch_size 300 --checkpoint_path runs/S2VT_VCR --feats_dir caption_data/feats/resnet152 --with_pop_visual 1 --with_selected_tool 1 --model S2VT_GCN_Sub --dim_vid 2048 --learning_rate 2e-4 
```
You may need to modify the path of "--input_json --info_json --caption_json --pop_visual_feats_dir --selected_tool_feats_dir" in the "opts.py" file if the $DATA_PATH is not setted as the default position.


## Evaluate trained models

    opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save/opt_info.json --saved_model runs/save/model_300.pth --batch_size 100 --gpu 0
```

Pretrained models can be downloaded from https://drive.google.com/file/d/11ewNRLsLMzFrPJxhGUWjDd4PxQukSYIy/view?usp=sharing
You may need to modify the paths in the opt_info.json file if the $DATA_PATH is not setted as the default position.

## Metrics

The evaluation is based on [coco-caption XgDuan](https://github.com/XgDuan/coco-caption/tree/python3). Thanks to port it to python3.


##  Optional Preprocess 

if you need to do preprocess for other videos and labels, you can refer to code below. It is not necessary if you want to do experiemnts described in the paper and have already downloaded the data we mentioned before.

```bash
python prepro_feats.py --output_dir data/feats/resnet152 --model resnet152

python prepro_vocab.py
```

## Acknowledgements
The code is built on top of the [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) and [video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch), we appreciate their efforts.
