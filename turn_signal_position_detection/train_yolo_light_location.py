from ultralytics import YOLO

def main():
    
    # 初始化模型（選擇你要的yolov版本）
    model = YOLO("yolo11m.pt")

    # 訓練模型
    model.train(
        data="dataset.yaml",     # dataset.yaml 與 dataset 資料夾同層
        epochs=100,              # 訓練輪數，可依需求調整
        imgsz=224,               # 輸入影像尺寸（會自動調整圖片為 640x640）
        batch=16,                # 每次處理的圖片數量，越大訓練越快，但需更大GPU記憶體
        name="yolo_light_train", # 訓練結果會儲存在 runs/detect/yolo_light_train/
        workers=2,               # 讀取資料用的 CPU 平行執行緒數
        verbose=True             # 顯示詳細訓練過程
    )

    # 驗證 val set
    metrics = model.val()  # 自動讀取 dataset.yaml 內 val 路徑
    print("驗證 mAP:", metrics.box.map)

    # 測試 test set
    metrics = model.val(split="test")  # 指定使用 test
    print("測試 mAP:", metrics.box.map)
    
if __name__ == '__main__':
    main()