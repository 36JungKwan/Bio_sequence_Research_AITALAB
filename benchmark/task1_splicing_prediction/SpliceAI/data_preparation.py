import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq
import os
import time
from tqdm import tqdm

# --- CẤU HÌNH ---
GENOME_PATH = r"D:\my_project\Bio_paper\Homo_sapiens.GRCh38.dna.primary_assembly.fa"
DATA_FOLDER = r"D:\my_project\Bio_paper\Bio_sequence_Research_AITALAB\train\task1_splicing_prediction\data_preparation\train_val"
PREPARED_FOLDER = "prepared_data/"
CONTEXT = 5000 

def get_sequence_worker(row, fasta_obj, target_len):
    try:
        parts = row['id'].split('_')
        chrom, pos, strand = parts[1], int(parts[2]), parts[3]
        label = int(row['Splicing_types'])
        
        # --- SỬA LỖI LỆCH TỌA ĐỘ (OFFSET FIX) ---
        # Chúng ta dịch chuyển vị trí thực tế sang trái 1bp để đưa GT, AG vào tâm.
        actual_pos = pos - 1
        
        # Chuyển từ 1-based (biographical) sang 0-based (python slice)
        start = (actual_pos - 1) - CONTEXT
        end = (actual_pos - 1) + CONTEXT + 1
        
        # Trích xuất trình tự
        seq_str = str(fasta_obj[chrom][max(0, start):end]).upper()
        
        # Bù 'N' nếu trình tự nằm ở biên nhiễm sắc thể
        if start < 0:
            seq_str = ("N" * abs(start)) + seq_str
        if len(seq_str) < target_len:
            seq_str = seq_str + ("N" * (target_len - len(seq_str)))
            
        # Reverse Complement nếu là mạch âm
        if strand == '-':
            seq_str = str(Seq(seq_str).reverse_complement())
            
        return seq_str
    except Exception as e:
        return "N" * target_len

def diagnose_splice_sites(df, sample_size=5):
    """Hàm kiểm tra xem cặp GT/AG đã nằm đúng vị trí trung tâm chưa"""
    print(f"\n{'Type':<10} | {'Window around center (-2 to +2)':<25} | {'Found?'}")
    print("-" * 65)
    
    for label, name in [(1, 'Donor'), (2, 'Acceptor')]:
        samples = df[df['Splicing_types'] == label]
        if len(samples) == 0: continue
        
        test_batch = samples.sample(min(sample_size, len(samples)))
        for _, row in test_batch.iterrows():
            seq = row['sequence']
            # Lấy 6 ký tự quanh tâm (vị trí 5000)
            # index 5000 và 5001 là nucleotide tại vị trí pos
            window = seq[4998:5004] 
            
            target = "GT" if label == 1 else "AG"
            # Đánh dấu vị trí 5000-5001 bằng dấu ngoặc []
            display_win = window[:2] + "[" + window[2:4] + "]" + window[4:]
            
            # Kiểm tra xem target có nằm đúng trong ngoặc không
            found = "✅" if window[2:4] == target else "❌"
            print(f"{name:<10} | {display_win:<25} | {found} (Target: {target})")

def prepare_csv_datasets(file_list):
    print(f"[{time.strftime('%H:%M:%S')}] Loading Genome with pyfaidx...")
    genome = Fasta(GENOME_PATH, sequence_always_upper=True)
    
    target_len = 2 * CONTEXT + 1
    os.makedirs(PREPARED_FOLDER, exist_ok=True)
    
    for file_name in file_list:
        file_start = time.time()
        input_path = os.path.join(DATA_FOLDER, file_name)
        output_path = os.path.join(PREPARED_FOLDER, file_name)
        
        if not os.path.exists(input_path):
            print(f"Skipping {file_name} (Not found)")
            continue
            
        df = pd.read_csv(input_path)
        
        # Sắp xếp để tối ưu hóa việc đọc file Genome từ ổ cứng
        print(f"Sorting {file_name} for sequential disk access...")
        df[['_tmp_chr', '_tmp_pos']] = df['id'].str.split('_', expand=True)[[1, 2]]
        df['_tmp_pos'] = df['_tmp_pos'].astype(int)
        df = df.sort_values(['_tmp_chr', '_tmp_pos']).reset_index(drop=True)

        print(f"🚀 Processing: {file_name} ({len(df)} rows)")
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            results.append(get_sequence_worker(row, genome, target_len))
            
        df['sequence'] = results
        
        # Chạy hàm chẩn đoán ngay sau khi trích xuất
        diagnose_splice_sites(df)
        
        # Lưu dữ liệu
        df = df.drop(columns=['_tmp_chr', '_tmp_pos'])
        df.to_csv(output_path, index=False)
        
        duration = time.time() - file_start
        print(f"✅ Saved to {output_path} | Speed: {len(df)/duration:.2f} seq/s\n")