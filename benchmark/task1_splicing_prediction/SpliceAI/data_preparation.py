import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- CẤU HÌNH ---
GENOME_PATH = r"D:\Homo_sapiens.GRCh38.dna.primary_assembly.fa"
DATA_FOLDER = r"D:\Bio_sequence_Research_AITALAB\train\task1_splicing_prediction\data_preparation\train_val"
PREPARED_FOLDER = "prepared_data/"
CONTEXT = 5000 

def get_sequence_worker(row, fasta_obj, target_len):
    try:
        parts = row['id'].split('_')
        chrom, pos, strand = parts[1], int(parts[2]), parts[3]
        
        start = (pos - 1) - CONTEXT
        end = (pos - 1) + CONTEXT + 1
        
        # pyfaidx lấy sequence cực nhanh và hỗ trợ xử lý biên tự động
        # Lưu ý: pyfaidx dùng 1-based indexing nhưng slice giống python
        seq_str = str(fasta_obj[chrom][max(0, start):end]).upper()
        
        # Bù 'N' nếu start âm
        if start < 0:
            seq_str = ("N" * abs(start)) + seq_str
        
        # Bù 'N' nếu thiếu độ dài
        if len(seq_str) < target_len:
            seq_str = seq_str + ("N" * (target_len - len(seq_str)))
            
        if strand == '-':
            seq_str = str(Seq(seq_str).reverse_complement())
            
        return seq_str
    except Exception:
        return "N" * target_len

def prepare_csv_datasets(file_list):
    # 1. Load Genome bằng pyfaidx (tạo file .fai để truy xuất cực nhanh)
    print(f"[{time.strftime('%H:%M:%S')}] Loading Genome with pyfaidx...")
    genome = Fasta(GENOME_PATH, sequence_always_upper=True)
    
    target_len = 2 * CONTEXT + 1
    os.makedirs(PREPARED_FOLDER, exist_ok=True)
    
    for file_name in file_list:
        file_start = time.time()
        input_path = os.path.join(DATA_FOLDER, file_name)
        output_path = os.path.join(PREPARED_FOLDER, file_name)
        
        df = pd.read_csv(input_path)
        
        # --- BÍ KÍP TĂNG TỐC: SORTING ---
        # Tách chrom và pos tạm thời để sort, giúp ổ cứng đọc tuần tự
        print(f"Sorting {file_name} for sequential disk access...")
        df[['_tmp_chr', '_tmp_pos']] = df['id'].str.split('_', expand=True)[[1, 2]]
        df['_tmp_pos'] = df['_tmp_pos'].astype(int)
        df = df.sort_values(['_tmp_chr', '_tmp_pos']).reset_index(drop=True)
        # -------------------------------

        print(f"🚀 Processing: {file_name} ({len(df)} rows)")
        
        # Dùng map hoặc list comprehension với pyfaidx thường nhanh hơn thread 
        # vì pyfaidx đã tối ưu việc buffer file.
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            results.append(get_sequence_worker(row, genome, target_len))
            
        df['sequence'] = results
        
        # Xóa các cột tạm và lưu
        df = df.drop(columns=['_tmp_chr', '_tmp_pos'])
        df.to_csv(output_path, index=False)
        
        duration = time.time() - file_start
        print(f"✅ Done {file_name} | Speed: {len(df)/duration:.2f} seq/s")