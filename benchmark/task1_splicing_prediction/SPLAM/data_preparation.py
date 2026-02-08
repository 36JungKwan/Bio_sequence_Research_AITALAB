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

# SPLAM sử dụng cửa sổ 800nt (400nt mỗi bên điểm nối)
CONTEXT = 400 
TARGET_LEN = 800 # Tổng độ dài đầu vào cho SPLAM

def get_sequence_worker(row, fasta_obj):
    try:
        parts = row['id'].split('_')
        chrom, pos, strand = parts[1], int(parts[2]), parts[3]
        label = int(row.get('Splicing_types', 0))
        
        # 1-based (CSV) sang 0-based (Python)
        pos0 = pos - 1 
        
        # Thiết lập Offset dựa trên loại site
        # Mục tiêu: Donor GT ở [400:402], Acceptor AG ở [398:400]
        if label == 1:  # DONOR
            offset = 400
        elif label == 2: # ACCEPTOR
            # Dịch thêm 2bp để đưa AG từ vị trí bị lệch về đúng index 398-399
            offset = 399
        else: # Nhãn 0 hoặc mặc định
            offset = 400

        if strand == '+':
            start = pos0 - offset
            end = start + 800
            seq = str(fasta_obj[chrom][max(0, start):end]).upper()
        else:
            # Mạch nghịch: Lấy vùng genome tương ứng rồi Reverse Complement
            # Để motif giữ đúng vị trí sau khi RC, ta lấy đối xứng qua tâm
            start = pos0 - (800 - offset - 1)
            end = pos0 + offset + 1
            seq = str(fasta_obj[chrom][max(0, start):end]).upper()
            seq = str(Seq(seq).reverse_complement())
            
        # Đảm bảo độ dài luôn là 800
        if len(seq) < 800:
            seq = seq.ljust(800, "N")
        else:
            seq = seq[:800]
            
        return seq
    except Exception as e:
        return "N" * 800

def diagnose_splice_sites(df, sample_size=5):
    print(f"\n{'Type':<10} | {'Sequence around Center (397-403)':<30} | {'Status'}")
    print("-" * 70)
    for label, name in [(1, 'Donor'), (2, 'Acceptor')]:
        samples = df[df['Splicing_types'] == label]
        if len(samples) == 0: continue
        for _, row in samples.sample(min(sample_size, len(samples))).iterrows():
            seq = row['sequence']
            # Donor: ...NN[GT]NN... (GT tại 400, 401)
            # Acceptor: ...[AG]NNNN... (AG tại 398, 399)
            if label == 1:
                part = seq[398:404]
                display = f"{part[:2]}[{part[2:4]}]{part[4:]}"
                found = "✅" if part[2:4] == "GT" else "❌"
            else:
                part = seq[396:402]
                display = f"{part[:2]}[{part[2:4]}]{part[4:]}"
                found = "✅" if part[2:4] == "AG" else "❌"
            print(f"{name:<10} | {display:<30} | {found}")

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