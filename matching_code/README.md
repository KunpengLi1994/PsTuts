# Text-Tutorial Clip Retrieval

## Requirements 
We recommended the following dependencies.

* Python 2.7 
* [PyTorch](http://pytorch.org/) (0.4.1)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data 

All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from google drive: https://drive.google.com/file/d/1cwml78PU7mGS5mcGJeIM0-q_j7HEIYcu/view?usp=sharing 

We refer to the path of extracted files for `data.zip` as `$DATA_PATH`. 

## Training new models
Run `train.py`:

```bash
python train.py --data_path $DATA_PATH --data_name coco_combine_precomp --logger_name runs/VCR --max_violation --img_dim 4135 --lr_update 30 --num_epochs 60
```

"coco" does not mean data from MSCOCO dataset, just means we follow MSCOCO dataset to organize our data.

## Evaluate trained models
Modify the model_path and data_path in the evaluation_models.py file. Then Run `evaluate.py`:

```bash
python evaluate.py
```
Pretrained models can be downloaded from https://drive.google.com/file/d/1RgqhYSG4yl_Ogazu1Kt2Awa7By5M7jco/view?usp=sharing 


## Acknowledgements
The code is built on top of the [VSE++](https://github.com/fartashf/vsepp), we appreciate their efforts. 

## Reference

If you found this code useful, please cite the following paper:

@inproceedings{li2020pstuts,
   author = {Li, Kunpeng and Fang, Chen and Wang, Zhaowen and Kim, Seokhwan and Jin, Hailin and Fu, Yun},
   title = {Screencast Tutorial Video Understanding},
   booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   year = {2020}
}

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


