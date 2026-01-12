# Hướng dẫn Chú giải Biến thể bằng Ensembl VEP (Docker)

File Notebook `vep_annotation_data.ipynb` được sử dụng để tự động hóa quá trình chú giải (annotation) dữ liệu biến thể di truyền từ file CSV sang các chỉ số chuyên sâu như **SpliceAI** và **dbNSFP** thông qua công cụ **Ensembl VEP**.

## 1. Yêu cầu hệ thống & Phần mềm

Trước khi bắt đầu, máy tính của bạn cần cài đặt các thành phần sau:

1. **Docker Desktop:**
* Tải và cài đặt tại: [docker.com](https://www.docker.com/products/docker-desktop/).
* **Lưu ý:** Đảm bảo Docker đang chạy khi bạn thực hiện chạy Notebook.


2. **Python Môi trường:**
* Cài đặt thư viện cần thiết: `pip install pandas`
* Trình chạy Notebook: Jupyter Lab hoặc VS Code (có cài extension Jupyter).


3. **Dung lượng ổ cứng:** Cần ít nhất **150GB - 200GB** trống nếu bạn định sử dụng đầy đủ plugin dbNSFP và SpliceAI.

---

## 2. Chuẩn bị Dữ liệu (Cực kỳ quan trọng)

VEP trong code này chạy ở chế độ **Offline** để đảm bảo tốc độ và tính riêng tư. Bạn cần chuẩn bị thư mục Cache theo cấu trúc sau:

### Cấu trúc thư mục mong muốn:

Bạn nên tạo một thư mục tên là `vep_cache` và sắp xếp như sau:

```text
vep_cache/
├── homo_sapiens/           # Dữ liệu cache chính của VEP (Version 110-115)
├── Plugins/                # Chứa file .pm của plugin
│   ├── SpliceAI.pm
│   └── dbNSFP.pm
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa      # File tham chiếu Genome
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai  # Index của file fa
├── spliceai_snv.vcf.gz     # Dữ liệu SpliceAI (kèm file .tbi)
├── spliceai_indel.vcf.gz   # Dữ liệu SpliceAI (kèm file .tbi)
└── dbNSFP.txt.gz           # Dữ liệu dbNSFP (kèm file .tbi)

```

### Cách tải nhanh dữ liệu:

* **VEP Cache:** Chạy lệnh `docker run -t -i -v /path/to/vep_cache:/opt/vep/.vep ensemblorg/ensembl-vep INSTALL.pl -a cf -s homo_sapiens -y GRCh38`
* **Plugins:** Tải từ [VEP_plugins GitHub](https://github.com/Ensembl/VEP_plugins).
* **Dữ liệu Plugin:** Tải từ các nguồn của Illumina (SpliceAI) và trang chủ dbNSFP. **Tất cả các file dữ liệu phải được Index bằng Tabix (tạo file .tbi).**

---

## 3. Cấu hình trong Notebook

Mở file `vep_annotation_data.ipynb` và tìm đến phần **CONFIG**. Bạn cần thay đổi các đường dẫn sao cho khớp với máy của bạn:
```python
# ==============================
# CONFIG
# ==============================
INPUT_CSV = r"đường/dẫn/đến/file_dau_vao.csv"
OUTPUT_CSV = r"ket_qua_annotation.csv"
CACHE_DIR = r"C:/Users/YourName/vep_cache" # Đường dẫn đến thư mục bạn vừa tạo ở bước 2
ASSEMBLY = "GRCh38"
# ==============================
```

---

## 4. Quy trình chạy Code

Notebook được chia làm 4 bước chính, bạn chỉ cần chạy lần lượt các ô (cells):

1. **Ô 1 (Đọc dữ liệu):** Load file CSV đầu vào. Đảm bảo file có ít nhất các cột: `CHROM`, `POS`, `REF`, `ALT`.
2. **Ô 2 (Tạo VCF & Chạy VEP):** * Code sẽ tự tạo một file `.vcf` tạm thời (vì VEP chỉ đọc file VCF).
* Lệnh Docker sẽ được khởi tạo, ánh xạ thư mục máy của bạn vào container.
* **Lưu ý:** Lần đầu chạy sẽ hơi chậm vì Docker cần tải hình ảnh `ensemblorg/ensembl-vep`.


3. **Ô 3 (Merge kết quả):** Đọc file kết quả từ VEP (`tmp_output.txt`) và gộp ngược lại vào dữ liệu CSV ban đầu của bạn dựa trên vị trí biến thể.
4. **Ô 4 (Kiểm tra):** Hiển thị các giá trị thiếu (null) để kiểm tra xem có bao nhiêu biến thể đã được chú giải thành công.

---

## 5. Các lỗi thường gặp

* **"Docker not found":** Đảm bảo bạn đã mở Docker Desktop.
* **"Permission denied":** Thư mục `CACHE_DIR` cần quyền đọc/ghi.
* **"File not found (.tbi)":** Mọi file dữ liệu lớn (`.gz`) của Plugin đều phải có file chỉ mục tương ứng đi kèm trong cùng thư mục.
* **Sai Assembly:** Nếu dữ liệu của bạn là hg19 (GRCh37) mà bạn cấu hình GRCh38, VEP sẽ không trả ra kết quả.
