import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq
import os
import time
from tqdm import tqdm

# --- CẤU HÌNH ---
GENOME_PATH = r"D:\Homo_sapiens.GRCh38.dna.primary_assembly.fa"
DATA_FOLDER = r"D:\Bio_sequence_Research_AITALAB\train\task1_splicing_prediction\data_preparation\train_val"
PREPARED_FOLDER = "prepared_data/"

# SPLAM sử dụng cửa sổ 800nt (400nt mỗi bên điểm nối)
CONTEXT = 400 
TARGET_LEN = 800 # Tổng độ dài đầu vào cho SPLAM

def get_sequence_worker(row, fasta_obj):
    try:
        parts = row['id'].split('_')
        # Giả định ID định dạng: label_chr_pos_strand (ví dụ: 1_chr1_12345_+)
        chrom, pos, strand = parts[1], int(parts[2]), parts[3]
        
        # Chuyển đổi từ 1-based sang 0-based
        # Trong sinh học, điểm Donor là nucleotide đầu tiên của Intron (G trong GT)
        # Điểm Acceptor là nucleotide cuối cùng của Intron (G trong AG)
        center_pos = pos - 1 
        
        # SPLAM lấy 400bp upstream và 400bp downstream
        # Window: [center-400 : center+400]
        start = center_pos - CONTEXT
        end = center_pos + CONTEXT
        
        # Trích xuất trình tự từ Genome
        seq_str = str(fasta_obj[chrom][max(0, start):end]).upper()
        
        # Bù 'N' nếu trình tự nằm ngoài biên nhiễm sắc thể
        if start < 0:
            seq_str = ("N" * abs(start)) + seq_str
        if len(seq_str) < TARGET_LEN:
            seq_str = seq_str + ("N" * (TARGET_LEN - len(seq_str)))
            
        # Reverse Complement nếu là mạch âm
        if strand == '-':
            seq_str = str(Seq(seq_str).reverse_complement())
            
        return seq_str
    except Exception as e:
        return "N" * TARGET_LEN

def diagnose_splice_sites(df, sample_size=5):
    """Kiểm tra xem GT/AG có nằm đúng vị trí trung tâm (index 400) không"""
    print(f"\n{'Type':<10} | {'Window at center (400)':<25} | {'Found?'}")
    print("-" * 65)
    
    for label, name in [(1, 'Donor'), (2, 'Acceptor')]:
        samples = df[df['Splicing_types'] == label]
        if len(samples) == 0: continue
        
        test_batch = samples.sample(min(sample_size, len(samples)))
        for _, row in test_batch.iterrows():
            seq = row['sequence']
            
            # SPLAM center là tại index 400. 
            # Donor (GT): GT bắt đầu tại 400, 401
            # Acceptor (AG): AG kết thúc tại 398, 399
            
            if label == 1: # Donor
                window = seq[398:404]
                target = "GT"
                display_win = window[:2] + "[" + window[2:4] + "]" + window[4:]
                found = "✅" if window[2:4] == target else "❌"
            else: # Acceptor
                window = seq[396:402]
                target = "AG"
                display_win = window[:2] + "[" + window[2:4] + "]" + window[4:]
                found = "✅" if window[2:4] == target else "❌"
                
            print(f"{name:<10} | {display_win:<25} | {found} (Target: {target})")

def prepare_csv_datasets(file_list):
    print(f"[{time.strftime('%H:%M:%S')}] Loading Genome...")
    genome = Fasta(GENOME_PATH, sequence_always_upper=True)
    os.makedirs(PREPARED_FOLDER, exist_ok=True)
    
    for file_name in file_list:
        file_start = time.time()
        input_path = os.path.join(DATA_FOLDER, file_name)
        output_path = os.path.join(PREPARED_FOLDER, file_name)
        
        if not os.path.exists(input_path): continue
            
        df = pd.read_csv(input_path)
        
        # Sắp xếp để đọc file nhanh hơn
        df[['_tmp_chr', '_tmp_pos']] = df['id'].str.split('_', expand=True)[[1, 2]]
        df['_tmp_pos'] = df['_tmp_pos'].astype(int)
        df = df.sort_values(['_tmp_chr', '_tmp_pos']).reset_index(drop=True)

        print(f"🚀 Processing: {file_name} ({len(df)} rows)")
        
        # Sử dụng list comprehension để tối ưu tốc độ
        df['sequence'] = [get_sequence_worker(row, genome) for _, row in tqdm(df.iterrows(), total=len(df))]
        
        # Kiểm tra tính đúng đắn của dữ liệu
        diagnose_splice_sites(df)
        
        # Lưu dữ liệu (Giữ nguyên các cột cũ và thêm cột sequence)
        df = df.drop(columns=['_tmp_chr', '_tmp_pos'])
        df.to_csv(output_path, index=False)
        
        duration = time.time() - file_start
        print(f"✅ Saved to {output_path} | Speed: {len(df)/duration:.2f} seq/s\n")