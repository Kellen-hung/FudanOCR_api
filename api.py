import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import io
import base64
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback # 引入 traceback 以便除錯
import json

# --- 引入你原本的專案模組 ---
from config import config
from util import get_alphabet, get_radical_alphabet, tensor2str
from model_loader import build_main_model, build_clip_model
from feature_extractor import extract_text_features

app = Flask(__name__)
# 允許跨域呼叫
CORS(app)

# 1. 讓中文正常顯示 (不要轉成 \uXXXX)
app.config['JSON_AS_ASCII'] = False
# 2. 讓回傳的 JSON 自動排版 (換行 + 縮排)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

class SimpleDigitCNN(nn.Module):
    def __init__(self):
        super(SimpleDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 輸出 0-9
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 全域變數：用來存放載入好的模型 ---
ocr_engine = {}

def init_model():
    """
    Server 啟動時執行一次，載入模型到記憶體中
    """
    print(f"正在載入模型 (使用裝置: {device})...")
    
    if not config.get('resume_model'):
        raise ValueError("[Error] Config 中的 'resume_model' 不能為空！")

    # 載入主模型
    model = build_main_model()
    model.eval()
    model = model.to(device)

    # 準備字典
    alphabet = get_alphabet()
    radical_alphabet = get_radical_alphabet()
    
    # 準備 CLIP
    clip_model = build_clip_model(len(radical_alphabet))
    text_features = extract_text_features(clip_model, radical_alphabet, config['alpha_path'])
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.to(device)

    # 存入全域變數
    ocr_engine['model'] = model
    ocr_engine['alphabet'] = alphabet
    ocr_engine['text_features'] = text_features
    
    try:
        digit_model = SimpleDigitCNN()
        # 載入剛剛訓練好的權重 (請確保 mnist_cnn.pth 在同一層目錄)
        digit_model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
        digit_model.eval()
        digit_model.to(device)
        ocr_engine['digit_model'] = digit_model
        print("成功載入 MNIST 手寫數字模型")
    except Exception as e:
        print(f"警告: 載入 MNIST 模型失敗 ({e})，數字辨識功能將無法使用。")
        ocr_engine['digit_model'] = None
    
    print("模型載入完成，API 準備就緒！")

# 程式啟動時載入模型
init_model()

# --- 新增/修改的工具函式 ---

def base64_to_pil(base64_str):
    """
    步驟 1: 將 Base64 字串還原為 PIL Image (完整大圖)
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Base64 Decode Error: {e}")
        return None

def preprocess_image_to_tensor(pil_image):
    """
    步驟 2: 將裁切後的 PIL Image 轉為模型可吃的 Tensor
    """
    # 讀取 config 設定
    h = config.get('img_h', 32) 
    w = config.get('img_w', 256)
    
    transform = transforms.Compose([
        transforms.Resize((h, w)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # [C, H, W] -> [1, C, H, W]
    img_tensor = transform(pil_image).unsqueeze(0)
    return img_tensor.to(device)

def run_inference(image_tensor):
    """
    執行單張圖片(Tensor)的推論
    """
    model = ocr_engine['model']
    alphabet = ocr_engine['alphabet']
    text_features = ocr_engine['text_features']
    
    max_length = config.get('char_len', 10)
    
    with torch.no_grad():
        batch_size = image_tensor.shape[0]
        pred = torch.zeros(batch_size, 1).long().to(device)
        image_features = None
        now_pred_indices = []

        # --- Autoregressive Decoding Loop ---
        for i in range(max_length):
            length_tmp = torch.zeros(batch_size).long().to(device) + i + 1
            
            result = model(
                image=image_tensor, 
                text_length=length_tmp, 
                text_input=pred, 
                conv_feature=image_features, 
                test=True
            )
            
            prediction = result['pred'][:, -1:, :].squeeze(1)
            prediction = prediction / prediction.norm(dim=1, keepdim=True)
            prediction = prediction @ text_features.t()
            
            now_token = torch.max(torch.softmax(prediction, 1), 1)[1]
            pred = torch.cat((pred, now_token.view(-1, 1)), 1)
            image_features = result['conv']
            
            token_val = now_token.item()
            if token_val == len(alphabet) - 1:
                break
                
            now_pred_indices.append(token_val)

        try:
            text_out = tensor2str(now_pred_indices)
        except Exception as e:
            text_out = ""
            
        return text_out

def preprocess_digit_image(pil_image):
    """ 
    數字模型的預處理 
    1. 轉灰階
    2. 反轉顏色 (白底黑字 -> 黑底白字)
    3. Resize 到 28x28 (MNIST 規格)
    """
    # 轉灰階
    gray_img = pil_image.convert('L')
    
    # 顏色反轉 (因為 MNIST 是黑底白字，但一般文件是白底黑字)
    # 這是提升準確率的關鍵！
    inverted_img = ImageOps.invert(gray_img)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(inverted_img).unsqueeze(0).to(device)

def run_digit_inference(pil_image):
    """ 數字模型推論 """
    model = ocr_engine.get('digit_model')
    if model is None:
        return ""
    
    with torch.no_grad():
        # 1. 預處理 (包含顏色反轉)
        input_tensor = preprocess_digit_image(pil_image)
        
        # 2. 推論
        output = model(input_tensor) # output shape: [1, 10]
        
        # 3. 取得最大機率的數字
        prediction = output.argmax(dim=1, keepdim=True) # 取得 index (就是數字 0-9)
        return str(prediction.item())

# --- API 路由設定 ---
@app.route('/ocr', methods=['POST'])
def handle_ocr():
    try:
        data = request.json
        
        # 1. 檢查輸入
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
            
        # 兼容 key 名稱 (image 或 image_base64)
        base64_str = data.get('image') or data.get('image_base64')
        if not base64_str:
            return jsonify({"status": "error", "message": "Missing 'image' field"}), 400
            
        if 'annotations' not in data:
            return jsonify({"status": "error", "message": "Missing 'annotations' field"}), 400
        
        # 2. 還原完整大圖
        full_image = base64_to_pil(base64_str)
        if full_image is None:
            return jsonify({"status": "error", "message": "Invalid base64 string"}), 400
            
        annotations = data['annotations']
        img_w, img_h = full_image.size
        
        ocr_results = []
        print(f"[Log] 收到請求，包含 {len(annotations)} 個區域，開始處理...")

        # 3. 遍歷切圖與辨識
        for i, ann in enumerate(annotations):
            ocr_texts = {
                'chinese': '',
                'digit': ''
            }
            if 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
                # x, y, w, h = ann['bbox']
                # left, top = int(x), int(y)
                # right, bottom = int(x + w), int(y + h)
                x1, y1, x2, y2 = ann['bbox']
                left, top = int(x1)+5, int(y1)+10
                right, bottom = int(x2)-5, int(y2)-10
                
                left = max(0, left)
                top = max(0, top)
                right = min(img_w, right)
                bottom = min(img_h, bottom)

                if right > left and bottom > top:
                    try:
                        crop_img = full_image.crop((left, top, right, bottom))
                        input_tensor = preprocess_image_to_tensor(crop_img)
                        ocr_texts['chinese'] = run_inference(input_tensor)
                        ocr_texts["digit"] = run_digit_inference(crop_img)
                    except Exception as inner_e:
                        print(f"Error processing bbox {i}: {inner_e}")
                        ocr_texts['chinese'] = ''
                        ocr_texts['digit'] = ''
            
            ocr_results.append(ocr_texts)

        # 4. 回傳結果 (不回傳圖片)
        response_data = {
            "status": "success",
            "ocr_results": ocr_results,
            "annotations": annotations  # 保留座標資訊方便比對
        }
        
        print(f"[Log] 處理完成，回傳 {len(ocr_results)} 筆結果。")
        
        json_str = json.dumps(response_data, ensure_ascii=False, indent=4)
        return app.response_class(json_str, mimetype='application/json')
        # return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        err_data = {"status": "error", "message": str(e)}
        return app.response_class(
            json.dumps(err_data, ensure_ascii=False, indent=4),
            mimetype='application/json',
            status=500
        )
        # return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # 啟動 Server，Port 設定為 8083
    app.run(host='0.0.0.0', port=8083, debug=False)