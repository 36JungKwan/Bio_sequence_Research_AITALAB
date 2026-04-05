# Interpretability - Task 1 Splicing Prediction

Notebook chinh: `interpretability_task1_splicing_prediction.ipynb`

## Muc tieu
- Chay 3 phuong phap interpretability cho pipeline: NT encoder + MLP head
- Phuong phap: Integrated Gradients, Attention Rollout, ISM
- Bao cao theo donor/acceptor va theo group `all/first/middle/last`

## Data schema toi thieu
CSV dau vao can cac cot:
- `site_id`
- `sequence` (do dai 601)
- `label` (0/1/2)
- `site_type` (`donor`/`acceptor`)
- `group_position` (`all`/`first`/`middle`/`last`)

## Chay notebook
1. Mo notebook va patch cac bien trong Section 1:
   - `INPUT_CSV_PATH`
   - `MODEL_CHECKPOINT_PATH`
2. Run All cells theo thu tu.
3. Kiem tra output trong:
   - `outputs/`
   - `plots/`
   - `artifacts/`

## Luu y khoa hoc
- Motif-only va motif-flank deu duoc bao cao trong notebook.
- Metric ket luan nen uu tien ISM (tinh nhan qua), ket hop IG/Rollout de kiem chung tinh nhat quan.
