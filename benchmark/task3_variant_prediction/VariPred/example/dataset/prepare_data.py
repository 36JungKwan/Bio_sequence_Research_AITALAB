import pandas as pd
import os

# Đọc file dữ liệu gốc của bạn
input_file = 'benchmark\task3_variant_prediction\VariPred\example\dataset\test.parquet' # Thay bằng tên file thực tế của bạn
df = pd.read_parquet(input_file)

def prepare_varipred_complete(df):
    varipred_df = pd.DataFrame()

    # 1. Xử lý Amino_acids: Không lọc bỏ, xử lý mọi trường hợp
    def get_aa_pair(aa_str):
        if pd.isna(aa_str) or str(aa_str).strip() == "":
            return "X", "X" # Fallback nếu dữ liệu trống
        
        parts = str(aa_str).split('/')
        if len(parts) == 2:
            return parts[0], parts[1] # Trường hợp E/G
        else:
            return parts[0], parts[0] # Trường hợp chỉ có 1 ký tự (đồng nghĩa)

    # Tách WT và MT acid amin
    aa_pairs = df['Amino_acids'].apply(get_aa_pair)
    varipred_df['wt_aa'] = [p[0] for p in aa_pairs]
    varipred_df['mt_aa'] = [p[1] for p in aa_pairs]

    # 2. target_id: Giữ AlleleID để đối chiếu
    varipred_df['target_id'] = df['AlleleID'].astype(str)

    # 3. aa_index: Vị trí trung tâm trong chuỗi con (0-based index)
    varipred_df['aa_index'] = df['prot_ref_seq'].apply(lambda x: len(str(x)) // 2)

    # 4. Sequences
    varipred_df['wt_seq'] = df['prot_ref_seq'].astype(str)
    varipred_df['mt_seq'] = df['prot_alt_seq'].astype(str)

    # 5. Label: Mặc định -1 để code VariPred chạy được
    varipred_df['label'] = -1

    return varipred_df

# Chuyển đổi toàn bộ không lọc
target_csv = prepare_varipred_complete(df)

# Xuất file vào thư mục VariPred
output_path = 'benchmark/task3_variant_prediction/VariPred/example/dataset/target.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
target_csv.to_csv(output_path, index=False)

print(f"--- Hoàn tất chuẩn bị dữ liệu ---")
print(f"Tổng số biến thể đưa vào inference: {len(target_csv)}")
print(f"File lưu tại: {output_path}")