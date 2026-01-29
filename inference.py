# inference.py
# CUDA_VISIBLE_DEVICES=GPU_NUM python inference.py
import torch
import torch.nn as nn
from config import config
from util import get_inference_data, get_alphabet, get_radical_alphabet, tensor2str, saver
from model_loader import build_main_model, build_clip_model
from feature_extractor import extract_text_features

def preprocessing(img_dir):
    # 準備模型
    if config['resume_model']:
        model = build_main_model()
        model.eval()
    else:
        print(f"\033[31m[Error] The \'resume model\' in config.py can not be empty!!! \033[0m")
        return
    
    # saver()
    # output_file = './history/{}/result.txt'.format(config['exp_name'])
        
    # 準備字典
    alphabet = get_alphabet()
    radical_alphabet = get_radical_alphabet()
    
    # 準備 CLIP text feature
    clip_model = build_clip_model(len(radical_alphabet))
    text_features = extract_text_features(clip_model, radical_alphabet, config['alpha_path'])
    
    # 讀 inference 資料
    dataloader = get_inference_data(img_dir)
    return model, alphabet, clip_model, text_features, dataloader

def inference(model, alphabet, clip_model, text_features, dataloader):
    
    total = 0
    results = []

    with torch.no_grad():
        for images, fnames in dataloader:
            images = images.cuda()
            batch = images.shape[0]
            # decoding
            max_length = config['char_len']
            pred = torch.zeros(batch, 1).long().cuda()
            image_features = None

            for i in range(max_length):
                length_tmp = torch.zeros(batch).long().cuda() + i + 1
                result = model(image=images, text_length=length_tmp, text_input=pred, conv_feature=image_features, test=True)
                prediction = result['pred'][:, -1:, :].squeeze()
                prediction = prediction / prediction.norm(dim=1, keepdim=True)
                prediction = prediction @ text_features.t()
                now_pred = torch.max(torch.softmax(prediction, 1), 1)[1]
                pred = torch.cat((pred, now_pred.view(-1, 1)), 1)
                image_features = result['conv']

            # decode tensor -> string
            for i in range(batch):
                now_pred = []
                for j in range(max_length):
                    if pred[i][j] != len(alphabet) - 1:  # skip END
                        now_pred.append(pred[i][j])
                    else:
                        break
                text_out = tensor2str(now_pred[1:])  # skip START
                results.append((fnames[i], text_out))
                print(f"{fnames[i]} -> {text_out}")
                total += 1
    return results
    # # 存結果
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for fname, pred in results:
    #         f.write(f"{fname}\t{pred}\n")

    # print(f"\033[35m[info] Done! Processed {total} images. Results saved to {output_file}\033[0m")

if __name__ == "__main__":
    model, alphabet, clip_model, text_features, dataloader = preprocessing(config['inference_dataset'])
    results = inference(model, alphabet, clip_model, text_features, dataloader)