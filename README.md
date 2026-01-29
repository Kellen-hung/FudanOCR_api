# The api version of FudanOCR

Download link of the models

### infomation
1. data/radical_all_Chinese_2.txt <-> models/adult_pre_train_model.pth
2. data/radical_all_Chinese.txt <-> models/author_pre_train_model.pth
3. final_model is for Chinese recognition, while mnist_cnn is for digital recognition
4. final_model will be updated after cram school completes the material preparation and fine-tuning.

### using miniconda
```bash=
conda env create -f myenv.yml
conda activate IDS
```

### after that, download pytorch in IDS environment
```bash=
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

### run api in the background
```bash=
nohup python api.py > ./logs/ocr.log 2>&1 &
```

### how to use?
```bash=
curl -X POST http://IP:port/ocr -H "Content-Type: application/json" -d @./test/input_2.json > ./test/output_2.json
```
