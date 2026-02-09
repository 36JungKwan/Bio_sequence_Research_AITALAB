import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq
import numpy as np

def get_splam_official_logic_seq(chrom, pos, strand, label, fasta_obj):
    """
    Mô phỏng logic của splam_extractor (C++):
    Ghép 400nt vùng Donor và 400nt vùng Acceptor.
    """
    pos0 = pos - 1
    # SPLAM trích xuất 200nt bên trái và 200nt bên phải site
    start, end = pos0 - 200, pos0 + 200
    
    # Lấy sequence từ FASTA (xử lý biên)
    chrom_len = len(fasta_obj[chrom])
    f_start, f_end = max(0, start), min(chrom_len, end)
    seq_flank = str(fasta_obj[chrom][f_start:f_end]).upper()
    
    # Padding 'N' nếu site nằm ở sát đầu/cuối chromosome
    seq_flank = ("N" * max(0, -start)) + seq_flank + ("N" * max(0, end - chrom_len))
    
    if strand == '-':
        seq_flank = str(Seq(seq_flank).reverse_complement())

    # Logic ghép chuỗi 800nt của SPLAM cho Site đơn lẻ:
    # [Phần Donor 400nt] + [Phần Acceptor 400nt]
    if label == 1 or label == 0: # Donor hoặc Negative
        # Site nằm ở giữa block đầu tiên (Index 200)
        return seq_flank + ("N" * 400)
    else: # Acceptor
        # Site nằm ở giữa block thứ hai (Index 600)
        return ("N" * 400) + seq_flank

def one_hot_encode_splam(seq):
    # Mapping chuẩn SPLAM: A:0, C:1, G:2, T:3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded.T # Trả về (4, 800)