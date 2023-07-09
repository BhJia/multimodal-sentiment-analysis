python train.py --train_data "data/train.txt" --data_folder "data/data" --text_model "roberta" --image_model "resnet" --pretrained_text "pretrained/sentiment-roberta" --pretrained_image "pretrained/resnet/resnet34.pth" --fusion_method "concat" --validation_size 0.2 --batch_size 4 --max_len 256 --cuda True --gpu "0" --seed 2 --epochs 10 --lr 2e-5 --weight_decay 1e-4 --save "EXP" --wandb_id ""