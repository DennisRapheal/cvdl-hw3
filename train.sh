

# python train.py --model_name resnet50_v2_baseline --model_type resnet50_v2 --epochs 100 --batch_size 4
# python train.py --model_name resnet50_baseline --model_type resnet50 --epochs 100 --batch_size 4




#alpha 30 sigma 12 # no transform
python train.py --model_name v1_b1 --model_type resnet50 --epochs 80 --batch_size 1

#alpha 30 sigma 12 # elastic
python train.py --model_name v1_b1_elastic --model_type resnet50 --epochs 80 --batch_size 1

#alpha 30 sigma 12
python train.py --model_name v2_b4_elastic --model_type resnet50_v2 --epochs 80 --batch_size 4
# python train.py --model_name resnet50_baseline --model_type resnet50 --epochs 100 --batch_size 4
# python train.py --model_name resnet50_v2_nocrop --model_type resnet50_v2 --epochs 70 --batch_size 4