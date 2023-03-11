# CPCR & CPRD


## Code Download
```
git clone https://github.com/Awenbocc/CPCR.git
```

## Shared Dataset For both Methods
(1) The Pre-training dataset can be downloaded from [here](https://drive.google.com/file/d/1vi1bMm_QX8rKdyug40MkG2GPZoAO_QCo/view?usp=sharing)

(2) Unzip it to ```datasets``` package


# CPCR
## Pre-training Step: CP

```
python pre-train/main.py \
  -a resnet50 \
  --lr 0.015 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos\
  --batch-size 128 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /dataset/crd
```

## Fine-tuning Step: CR 
#### `2023/03/01`: code checking, coming soon...
### For VQA-RAD:
```
bash med-vqa/vqa_rad.sh
```
### For Slake:
```
bash med-vqa/slake.sh
```


# CPRD

