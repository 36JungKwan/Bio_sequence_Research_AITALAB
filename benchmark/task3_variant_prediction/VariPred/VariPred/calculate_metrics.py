import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    balanced_accuracy_score, 
    f1_score, 
    matthews_corrcoef, 
    precision_score, 
    recall_score
)

def evaluate_performance(res_file, original_file):
    # 1. Đọc kết quả dự đoán (Dùng tab separator)
    results = pd.read_csv(res_file, sep='\t')
    # 2. Đọc file gốc có NHÃN THẬT (không dùng file target.csv nhãn -1)
    original = pd.read_csv(original_file)
    
    # 2. Merge dữ liệu để khớp nhãn và điểm số
    # Đảm bảo AlleleID cùng kiểu dữ liệu
    results['AlleleID'] = results['target_id'].astype(int)
    original['AlleleID'] = original['AlleleID'].astype(int)
    
    df = pd.merge(results, original[['AlleleID', 'ClinicalSignificance']], on='AlleleID')
    
    label_map = {
        'Pathogenic': 1,
        'Benign': 0,
    }

    # 3. Tiền xử lý nhãn (Chỉ lấy các dòng có nhãn 0 hoặc 1)
    df['label'] = df['ClinicalSignificance'].map(label_map)
    df['label'] = df['label'].astype(int)
    
    y_true = df['label'].values
    y_scores = df['prediction_score'].values
    
    # Chuyển đổi xác suất thành nhãn nhị phân (ngưỡng 0.5)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # 4. Tính toán các Metric theo yêu cầu của bạn
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_scores),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }
    
    return metrics

# Thực hiện tính toán
if __name__ == "__main__":
    # Thay tên file bằng file thực tế của bạn
    res_file = 'benchmark\task3_variant_prediction\VariPred\example\output_results\VariPred_output.txt' 
    ori_file = 'benchmark\task3_variant_prediction\VariPred\example\dataset\test.parquet'
    
    try:
        results_dict = evaluate_performance(res_file, ori_file)
        
        # Xuất kết quả theo định dạng bạn yêu cầu
        print(json.dumps(results_dict, indent=4))
        
        # Lưu kết quả vào file json nếu cần
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
            
    except Exception as e:
        print(f"Lỗi: {e}. Hãy đảm bảo bạn đã chạy inference và có file kết quả.")