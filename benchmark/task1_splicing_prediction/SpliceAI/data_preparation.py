import pandas as pd
import pysam
import os

# --- CẤU HÌNH PREPARATION ---
GENOME_PATH = "path/to/hg38.fa"
DATA_FOLDER = "path/to/raw_data/"
PREPARED_FOLDER = "prepared_data/"
CONTEXT = 5000 

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq.upper()))

def prepare_csv_datasets(file_list):
    fasta = pysam.FastaFile(GENOME_PATH)
    target_len = 2 * CONTEXT + 1 # Độ dài chuẩn cần thiết
    
    for file_name in file_list:
        input_path = os.path.join(DATA_FOLDER, file_name)
        output_path = os.path.join(PREPARED_FOLDER, file_name)
        
        print(f"Processing: {file_name}...")
        df = pd.read_csv(input_path)
        
        new_sequences = []
        for idx, row in df.iterrows():
            parts = row['id'].split('_')
            chrom, pos, strand = parts[1], int(parts[2]), parts[3]
            
            # Giải quyết Vấn đề 4: Tọa độ âm
            start = (pos - 1) - CONTEXT
            end = (pos - 1) + CONTEXT + 1
            
            actual_start = max(0, start)
            
            seq = fasta.fetch(chrom, actual_start, end)
            
            # Bù 'N' nếu start < 0 (đầu nhiễm sắc thể)
            if start < 0:
                seq = ("N" * abs(start)) + seq
            
            # Bù 'N' nếu seq ngắn hơn yêu cầu (cuối nhiễm sắc thể)
            if len(seq) < target_len:
                seq = seq + ("N" * (target_len - len(seq)))
            
            if strand == '-':
                seq = reverse_complement(seq)
                
            new_sequences.append(seq.upper())
        
        df['sequence'] = new_sequences
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")