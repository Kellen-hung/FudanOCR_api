# feature_extractor.py
import torch
from util import convert_char

def extract_text_features(clip_model, radical_alphabet, alpha_path):
    char_file = open(alpha_path, 'r').read()
    chars = list(char_file)
    tmp_text = convert_char(chars)

    text_features = [torch.zeros([1, 2048]).cuda()]
    iters = len(chars) // 100

    with torch.no_grad():
        for i in range(iters + 1):
            s, e = i * 100, (i + 1) * 100
            e = min(e, len(chars))
            text_features_tmp = clip_model.module.encode_text(tmp_text[s:e])
            text_features.append(text_features_tmp)

        text_features.append(torch.ones([1, 2048]).cuda())
        text_features = torch.cat(text_features, dim=0).detach()

    return text_features
