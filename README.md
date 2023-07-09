# Multimodal-Sentiment-Analysis

This is the official repository of the fifth assignment of the *'Contemporary Artificial Intelligence'* . This project including multimodal sentiment analysis using **RoBERTa** and **ResNet**



### Repository structure

```python
│  dataset.py		# dataset
│  dataloader.py	# dataloader
│  model.py			# model definition
│  README.md
│  requirements.txt		# packages required
│  test.py		# testing file 
│  test.sh		# testing script
│  train.py		# multimodal model training file
│  train.sh		# multimodal model training script
│  train_image.py		# image model training file(ablation study)
│  train_text.sh		# text model training file(ablation study)
│  utils.py				# some utility definitions and functions
├─checkpoints			# best model weights
│      best_model.pth
├─data				# training and testing data
       ├─data		# training data	    
│      train.txt	# training labels data
│      test_without_label.txt	# testing data   
├─pretrained	# pretrained model
       ├─resnet
       ├─sentiment-roberta
├─results	# testing results
│      output.txt
```



### Installation

environment: linux(Ubuntu20.04) + CUDA 11.7 / Windows10 + CUDA 11.7

1.create a new conda environment

```
conda create -n multimodal python=3.7
conda activate multimodal
```

2.install **requirements.txt** in a **python>=3.7** environment

```python
pip install -r requirements.txt  
```

**Note:** torch version should be carefully chosen according to the CUDA version.



### Testing

1.put best model weights in `checkpoints`

best model weights can be downloaded from BaiduNetdisk

https://pan.baidu.com/s/127bMc91D3O8YZ53V0cADug?pwd=aykg 

2.put testing data in `data`

3.run following command

```python
python test.py --test_data "data/test_without_label.txt" --data_folder "data/data" --text_model "roberta" --image_model "resnet" --pretrained_text "pretrained/sentiment-roberta" --pretrained_image "pretrained/resnet/resnet34.pth" --fusion_method "concat" --weights "checkpoints/best_model.pth" --batch_size 4 --max_len 256 --cuda True --gpu "0" --seed 2 --epochs 10 --save_path "results" 
```

or run `test.sh` (linux)

```
sh test.sh
```



### Training

1.put pretrained models in  `pretrained` 

**Note:** put pretrained RoBERTa model in `pretrained/sentiment-roberta`, put pretrained ResNet model in `pretrained/resnet`

① pretrained RoBERTa model can be downloaded from [sentiment-roberta-large-english-3-classes](https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes)

② ResNet model can be downloaded from [resnet18](https://download.pytorch.org/models/resnet18-f37072fd.pth), [resnet34](https://download.pytorch.org/models/resnet34-b627a593.pth), [resnet50](https://download.pytorch.org/models/resnet50-0676ba61.pth)

2.put training data in `data`

3.run following command

```python
python train.py --train_data "data/train.txt" --data_folder "data/data" --text_model "roberta" --image_model "resnet" --pretrained_text "pretrained/sentiment-roberta" --pretrained_image "pretrained/resnet/resnet34.pth" --fusion_method "concat" --validation_size 0.2 --batch_size 4 --max_len 256 --cuda True --gpu "0" --seed 2 --epochs 10 --lr 2e-5 --weight_decay 1e-4 --save "EXP" --wandb_id ""
```

or run `train.sh`(linux)

```
sh train.sh
```



**Other image models**

change `pretrained_image` args in `train.py` or `train.sh` to `pretrained/resnet/resnet18.pth` or `pretrained/resnet/resnet50.pth` 



**Other fusion methods**

change `fusion_methods` args in `train.py` or `train.sh` to `add` or `attention`



### Attribution

Parts of this code referenced the following repositories:

- [huggingface/transformers](https://github.com/huggingface/transformers)

- [WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

