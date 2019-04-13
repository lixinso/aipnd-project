# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


python train.py --device cuda --model_type densenet --checkpoint_path densenet121_checkpoint.pth --top_k 5 --category_names_path cat_to_name.json --learning_rate 0.001 --hidden_units 500 --epochs 1

python predict.py --img_path ./flowers/train/102/image_08001.jpg --device cuda --model_type densenet --checkpoint_path densenet121_checkpoint.pth --top_k 5 --category_names_path cat_to_name.json
