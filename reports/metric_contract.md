# Metric contract — T02 Raw QC Triage

| Mục | Cam kết của nhóm |
|---|---|
| Track | T02 — Raw QC Triage |
| Primary metric | Defect Recall@B và Precision@B |
| Tăng tốt hay giảm tốt? | Higher is better |
| Đơn vị đánh giá | Image |
| Budget B | 20 |
| Candidate pool | 96 fixture images from `fixtures/t02_raw_qc/images.csv` |
| Eval split | `eval` split; không tune trên eval |
| Dev split | `dev` split; dùng để calibrate phân phối sharpness |
| Seed | 42 |
| Prediction file | `out/<run>/ranked_review.csv` |
| Prediction columns | `image_id,score` |
| Score direction | Score cao hơn = nghi lỗi hơn, ưu tiên review trước |
| Recall formula | `# defective images in top-B / # defective images in eval` |
| Precision formula | `# defective images in top-B / B` |
| Threshold policy | Score weights chốt trước khi xem GT: blur 0.55, dark 0.25, glare 0.20; review queue B=20 |
| Slices | `light`: day/dusk/night; `camera_id`: cam_A/cam_B |
| Error costs | Missed defect có rủi ro cao hơn false alarm; false alarm làm tăng review/recapture cost |
| Uncertainty | Dùng bootstrap/Wilson theo evaluator khi có GT; hiện chưa có GT overlay |
| GT status | Chờ instructor overlay; không tự tạo nhãn hoặc tự điền metric |
| Owner | Nguyễn Quang Thức — Lead / Integrator |
| Review owner | Nguyễn Quốc Thắng — `Thang-03158` |
| Freeze | 2026-11-14, Group G029 |

## Evaluation commands

```powershell
.venv\Scripts\python.exe -m src.evaluate --track t02 --gt fixtures/t02_raw_qc/gt/labels.csv --pred out/baseline/ranked_review.csv --run-name baseline --out reports/metrics_baseline.json
.venv\Scripts\python.exe -m src.evaluate --track t02 --gt fixtures/t02_raw_qc/gt/labels.csv --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json
.venv\Scripts\python.exe -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
```

Các lệnh trên chỉ chạy được khi giảng viên cung cấp overlay `gt/`. Không thay đổi định nghĩa metric để làm đẹp kết quả.
