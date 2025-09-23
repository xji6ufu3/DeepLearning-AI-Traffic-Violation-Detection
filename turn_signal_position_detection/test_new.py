import os
import cv2
from ultralytics import YOLO
import re
import pandas as pd
from tqdm import tqdm
import sys

draw_bbox = 1

if draw_bbox:
    print(f"繪製帶有 bbox 的畫面")

# 路徑設定
# INPUT_DIR = "../transformer/data/test_police"
INPUT_DIR = "../transformer/data/test_light_way"
OUTPUT_DIR = "../transformer/light_position/test_light_way"
WEIGHTS_PATH = "runs/detect/yolo_light_train2/weights/best.pt"
OUTPUT_WITH_BOX_DIR = "output/with_bbox/test_light_way"

print(f'INPUT_DIR: {INPUT_DIR}')
print(f'OUTPUT_DIR: {OUTPUT_DIR}')
print(f'WEIGHTS_PATH: {WEIGHTS_PATH}')
print(f'OUTPUT_WITH_BOX_DIR: {OUTPUT_WITH_BOX_DIR}')

# 建立輸出資料夾（如果還沒存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_WITH_BOX_DIR, exist_ok=True)

# os.makedirs(OUTPUT_WITH_BOX, exist_ok=True)
# os.makedirs(OUTPUT_ONLY_BOX, exist_ok=True)

# 載入模型
model = YOLO(WEIGHTS_PATH)

# 類別名稱
class_names = ['l', 'r']

# 讀取 input 資料夾內所有圖片
# image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 遍歷每個子資料夾
car_folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]

bbox_num = 0

xlsx_file = os.path.join('../transformer', '警方的未使用方向燈舉發資料.xlsx')
df = pd.read_excel(xlsx_file)

for filename in tqdm(os.listdir(INPUT_DIR), desc="YOLO 判斷方向燈位置"):
    
    if filename.startswith('.'):  # 濾除隱藏資料夾 (.ipynb_checkpoints)
        continue
    
    input_folder = os.path.join(INPUT_DIR, filename)
    output_folder = os.path.join(OUTPUT_DIR, filename)
    output_with_bbox_folder = os.path.join(OUTPUT_WITH_BOX_DIR, filename)
    
    if not os.path.isdir(input_folder):
        continue  # 忽略非資料夾
        
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_with_bbox_folder, exist_ok=True)
        
    row_match = df[df['影片名稱'] == filename]
    
    if row_match.empty:
        print(f"[警告] 找不到影片名稱: {filename}")
        break

    # begin = 1
    # end = len([f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')])
    turn = row_match.iloc[0]['轉向']

    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')],
        key=lambda x: int(re.search(r'car\d+_(\d+)', x).group(1))
    )
    
    for img_name in image_files:
        
        img_path = os.path.join(input_folder, img_name)
        
        if not os.path.exists(img_path):
            print(f"[跳過] 圖片不存在: {img_path}")
            continue

        img = cv2.imread(img_path)
        results = model(img, verbose=False)[0]

        if len(results.boxes) == 0:
            print(f"no bbox: {filename}/{img_name}")
            continue

        '''
        # 用 x_center 分辨左右
        boxes_with_center = []
        for box in results.boxes:
            x_center = box.xywh[0][0].item()
            boxes_with_center.append((x_center, box))

        boxes_with_center.sort(key=lambda x: x[0])
        '''
        
        # 用與畫面中心的歐式距離挑選最近的兩個 bbox
        h, w, _ = img.shape
        center_x, center_y = w / 2, h / 2

        boxes_with_distance = []
        for box in results.boxes:
            x_center, y_center = box.xywh[0][0].item(), box.xywh[0][1].item()
            distance = ((x_center - center_x) ** 2 + (y_center - center_y) ** 2) ** 0.5
            boxes_with_distance.append((distance, box, x_center))  # 附帶 x_center 供後續判斷左右

        boxes_with_distance.sort(key=lambda x: x[0])  # 按距離排序
        
        if len(boxes_with_distance) == 1:
            light_assignments = [(turn, boxes_with_distance[0][1])]
        else:
            # light_assignments = [
            #     ('l', boxes_with_center[0][1]),
            #     ('r', boxes_with_center[-1][1])
            # ]
            # 挑前兩名，再根據 x_center 左右指派
            two_boxes = sorted(boxes_with_distance[:2], key=lambda x: x[2])  # x[2] 是 x_center
            light_assignments = [
                ('l', two_boxes[0][1]),
                ('r', two_boxes[1][1])
            ]
            
        if draw_bbox:
            img_with_boxes = img.copy()
        
        car_str = None
        frame_str = None
        
        for light_type, box in light_assignments:
            if light_type != turn:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w, _ = img.shape
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            cropped = img[y1:y2, x1:x2]

            match = re.search(r'car(\d+)_(\d+)', img_name)
            if not match:
                print(f"[跳過] 無法匹配 car_id/frame_id: {img_name}")
                continue
            car_str = match.group(1)
            frame_str = match.group(2)
            output_path = os.path.join(output_folder, f"car{car_str}_{frame_str}.jpg")
            cv2.imwrite(output_path, cropped)

            #####======== 繪製帶有 bbox 的畫面 ========#####
            
            if draw_bbox:
                # 顏色依照左右可自定
                color = (0, 0, 255) if light_type == 'l' else (255, 0, 0)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)  # 框線粗細為 2 像素
                cv2.putText(img_with_boxes, light_type.upper(), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                '''
                img_with_boxes：目標圖片。
                light_type.upper()：顯示的文字內容，會將 'l' 或 'r' 轉成 'L' 或 'R'。
                (x1, y1 - 5)：文字的位置，比框的左上角高一點（避免重疊）。
                cv2.FONT_HERSHEY_SIMPLEX：使用的字體。
                0.5：字體大小。
                color：文字顏色（同 bbox 顏色）。
                2：文字線條粗細。
                '''
                
        if draw_bbox and car_str is not None and frame_str is not None:
            output_with_box_path = os.path.join(output_with_bbox_folder, f"car{car_str}_{frame_str}.jpg")
            cv2.imwrite(output_with_box_path, img_with_boxes)
    
    print(f"saved: {filename}")

print(f"所有訓練圖片已處理完畢，結果輸出至 {OUTPUT_DIR}")