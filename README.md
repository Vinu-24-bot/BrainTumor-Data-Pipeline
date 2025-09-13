# BrainTumor-Data-Pipeline
Automated MRI ingestion &amp; preprocessing workflows (DICOM → Cloud Storage → Dataflow → BigQuery).

# BrainTumor Data Pipeline

End‑to‑end MRI workflow: DICOM → Cloud Storage → Dataflow (Apache Beam) → BigQuery, plus an interactive Streamlit app for region‑growing segmentation and insights. Offloading preprocessing to Dataflow accelerates app‑side exploration and can significantly reduce per‑image processing time.

---

## Features

- Automated ingestion: upload local DICOMs to Cloud Storage with a simple helper.
- Scalable preprocessing: Dataflow pipeline windows, normalizes, and enhances DICOM slices; writes PNGs back to GCS.
- Structured analytics: per‑study metadata and image stats streamed to BigQuery via Storage Write API.
- Interactive segmentation app: load preprocessed images from GCS or upload files, run region growing, and view quality metrics (Dice, Jaccard, SSIM, BF Score).
- Modular Python stack: OpenCV, SimpleITK, pydicom, Streamlit, Apache Beam, GCS, BigQuery.

---

## Architecture

Local DICOMs
│
├── (gcp_utils.py) → Upload → gs://<RAW_BUCKET>/raw/.dcm
│
└── (Dataflow: dataflow_pipeline.py)
├─ Read DICOMs from GCS
├─ Window + CLAHE + smooth
├─ Write PNG → gs://<PROC_BUCKET>/processed/.png
└─ Stream metadata/stats → BigQuery <DATASET>.<TABLE>

Streamlit App (app.py)
├─ Option 1: Upload a file
└─ Option 2: Browse preprocessed PNGs from GCS → fast visualization
└─ Region Growing + Metrics (Dice, Jaccard, SSIM, BF Score)

text

---

## Repository Structure

.
├─ app.py # Streamlit UI
├─ dataflow_pipeline.py # Apache Beam/Dataflow pipeline
├─ gcp_utils.py # GCS upload/list/download helpers
├─ bq_schema.json # BigQuery table schema
├─ region_growing.py # Region growing segmentation
├─ evaluation.py # Area, perimeter, sensitivity/specificity, Dice, Jaccard
├─ advanced_metrics.py # SSIM, BF Score, radar plot
├─ preprocessor.py # Local image loaders + helpers
├─ utils.py # Preprocessing, overlay, probabilities
├─ requirements.txt # PIP dependencies
└─ pyproject.toml # Project metadata (optional)

text

---

## Prerequisites

- Python 3.11
- Google Cloud CLI (`gcloud`) and a GCP project
- Enable APIs: Cloud Storage, Dataflow, BigQuery
- IAM: a service account (or user) with roles:
  - Storage Object Admin
  - Dataflow Developer (and Worker for runner)
  - BigQuery Data Editor (or a more restrictive table‑level role)

Authenticate:

gcloud auth login
gcloud auth application-default login
gcloud config set project <PROJECT_ID>

text

---

## Installation

clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

install
python -m venv .venv && source .venv/bin/activate # (Linux/macOS)

or: .venv\Scripts\activate (Windows)
pip install -r requirements.txt

text

---

## Cloud Setup

export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=<YOUR_REGION> # e.g., us-central1, asia-south1
export RAW_BUCKET=<YOUR_RAW_BUCKET> # must already exist
export PROC_BUCKET=<YOUR_PROCESSED_BUCKET>
export DATASET=<YOUR_BQ_DATASET> # e.g., mri
export TABLE=<YOUR_BQ_TABLE> # e.g., mri_ingest

text

Create resources (if needed):

gcloud storage buckets create gs://$RAW_BUCKET --location=$REGION
gcloud storage buckets create gs://$PROC_BUCKET --location=$REGION
bq --location=$REGION mk -d $PROJECT_ID:$DATASET

Table can be auto‑created by the pipeline using bq_schema.json
text

---

## Ingest DICOMs → Cloud Storage

python -c "from gcp_utils import upload_dir_to_gcs;
print(upload_dir_to_gcs('path/to/local/dicoms', '$RAW_BUCKET', 'raw'))"

text

Your DICOMs should appear under `gs://$RAW_BUCKET/raw/`.

---

## Run the Dataflow Pipeline

Local (DirectRunner) quick test:

python dataflow_pipeline.py
--input gs://$RAW_BUCKET/raw/*.dcm
--output_prefix gs://$PROC_BUCKET/processed
--bq_table $PROJECT_ID:$DATASET.$TABLE
--schema_file bq_schema.json
--runner DirectRunner

text

Managed (DataflowRunner) at scale:

python dataflow_pipeline.py
--input gs://$RAW_BUCKET/raw/*.dcm
--output_prefix gs://$PROC_BUCKET/processed
--bq_table $PROJECT_ID:$DATASET.$TABLE
--schema_file bq_schema.json
--runner DataflowRunner
--project $PROJECT_ID
--region $REGION
--temp_location gs://$RAW_BUCKET/temp

text

Outputs:
- Processed PNGs in `gs://$PROC_BUCKET/processed/`
- A BigQuery table `<PROJECT_ID>.<DATASET>.<TABLE>` with one row per processed file  
  (columns defined in `bq_schema.json`).

---

## Explore BigQuery (examples)

-- Average intensity per study
SELECT study_uid, AVG(mean_intensity) AS avg_intensity
FROM PROJECT_ID.DATASET.TABLE
GROUP BY study_uid
ORDER BY avg_intensity DESC
LIMIT 50;

-- Latest processed files
SELECT file_name, gcs_uri_processed, processed_at
FROM PROJECT_ID.DATASET.TABLE
ORDER BY processed_at DESC
LIMIT 20;

text

---

## Run the Streamlit App

streamlit run app.py

text

Inside the app:
- Choose “From GCS (preprocessed)”
  - Enter bucket (e.g., `$PROC_BUCKET`) and prefix (e.g., `processed/`)
  - Select a PNG and run segmentation instantly on the preprocessed image
- Or choose “Upload” to use local files

The app shows:
- Segmentation overlay and comparison (if ground truth provided)
- Metrics: Area, Perimeter, Circularity, Dice, Jaccard, SSIM, BF Score, Sensitivity, Specificity
- Histograms and a radar chart for quick quality insight

---

## CLI/Code Snippets

Upload a single image to test app quickly:

from gcp_utils import list_images, download_image_np
names = list_images("<PROC_BUCKET>", "processed/", limit=5)
img = download_image_np("<PROC_BUCKET>", names)
print(img.shape)

text

---

## Configuration Notes

- `bq_schema.json` controls the BigQuery schema; the pipeline will create the table if it does not exist.
- `dataflow_pipeline.py` uses windowing + CLAHE + smoothing and writes PNG previews; adapt transforms as needed.
- The Streamlit app remains compatible with local uploads; the GCS option is purely additive.

---

## Troubleshooting

- Permission errors on Dataflow/BigQuery/Storage:
  - Confirm API enablement and IAM roles; re‑run `gcloud auth application-default login`.
- “Table not found”:
  - Ensure dataset exists (`bq mk -d`) or let the pipeline auto‑create the table with the provided schema.
- “Insufficient permissions to write to GCS”:
  - Verify the bucket name/region and that the service account has Storage permissions.
- App can’t list images:
  - Check bucket/prefix, network access, and that PNGs exist in `processed/`.

---

## Tech Stack

- Python, OpenCV, SimpleITK, pydicom, Streamlit
- Apache Beam (Dataflow), Google Cloud Storage, BigQuery

---

## Roadmap

- Add Pub/Sub‑triggered streaming ingestion
- Batch series reconstruction and 3D segmentation
- Model‑based tumor detection (baseline UNet) alongside region growing

---

## License

MIT (or choose a license and update this section)

---

## Acknowledgements

- Open source libraries: Apache Beam, OpenCV, SimpleITK, pydicom, Streamlit
- Google Cloud platform services: Cloud Storage, Dataflow, BigQuery
