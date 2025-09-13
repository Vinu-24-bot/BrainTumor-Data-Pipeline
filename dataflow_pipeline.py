import argparse
import io
import os
from datetime import datetime, timezone
from hashlib import sha256

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.io.gcp.internal.clients import bigquery as bq_types

import numpy as np
import cv2
import pydicom


def _apply_windowing(arr, ds):
    wc = None
    ww = None
    try:
        wc_val = ds.get("WindowCenter", None)
        ww_val = ds.get("WindowWidth", None)
        wc = float(wc_val[0] if isinstance(wc_val, pydicom.multival.MultiValue) else wc_val)
        ww = float(ww_val[0] if isinstance(ww_val, pydicom.multival.MultiValue) else ww_val)
    except Exception:
        wc, ww = None, None

    if wc is not None and ww is not None and ww > 0:
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        arr = np.clip(arr.astype(np.float32), low, high)
        arr = ((arr - low) / max(high - low, 1e-6) * 255.0).astype(np.uint8)
        return arr, wc, ww

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255.0
    return arr.astype(np.uint8), None, None


def _enhance_for_export(img_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img_u8)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.5)
    return img


def _table_schema_from_json(json_dict):
    schema = bq_types.TableSchema()
    for f in json_dict["fields"]:
        field = bq_types.TableFieldSchema()
        field.name = f["name"]
        field.type = f["type"]
        field.mode = f["mode"]
        schema.fields.append(field)
    return schema


class ProcessDicomDoFn(beam.DoFn):
    def __init__(self, processed_prefix_gs, table_schema_json):
        self.processed_prefix_gs = processed_prefix_gs.rstrip("/")
        self.schema_json = table_schema_json

    def setup(self):
        self.table_schema = _table_schema_from_json(self.schema_json)

    def process(self, readable_file: fileio.ReadableFile):
        file_path = readable_file.metadata.path
        file_name = os.path.basename(file_path)
        with readable_file.open() as f:
            ds = pydicom.dcmread(f, force=True)
            arr = ds.pixel_array

        img_u8, wc, ww = _apply_windowing(arr, ds)
        img_u8 = _enhance_for_export(img_u8)

        ok, buf = cv2.imencode(".png", img_u8)
        if not ok:
            return
        png_bytes = buf.tobytes()

        out_uri = f"{self.processed_prefix_gs}/{os.path.splitext(file_name)[0]}.png"
        with FileSystems.create(out_uri, mime_type="image/png") as w:
            w.write(png_bytes)

        mean_val = float(np.mean(img_u8))
        std_val = float(np.std(img_u8))

        pid = str(getattr(ds, "PatientID", ""))
        patient_hash = sha256(pid.encode("utf-8")).hexdigest()[:16] if pid else None

        row = {
            "file_name": file_name,
            "gcs_uri_raw": file_path,
            "gcs_uri_processed": out_uri,
            "patient_id_hash": patient_hash,
            "study_uid": str(getattr(ds, "StudyInstanceUID", "")) or None,
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")) or None,
            "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")) or None,
            "modality": str(getattr(ds, "Modality", "")) or None,
            "rows": int(img_u8.shape[0]),
            "cols": int(img_u8.shape[1]),
            "mean_intensity": mean_val,
            "std_intensity": std_val,
            "window_center": float(wc) if wc is not None else None,
            "window_width": float(ww) if ww is not None else None,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        yield row


def run(argv=None):
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="gs://bucket/raw/*.dcm")
    parser.add_argument("--output_prefix", required=True, help="gs://bucket/processed")
    parser.add_argument("--bq_table", required=True, help="PROJECT:DATASET.TABLE")
    parser.add_argument("--schema_file", default="bq_schema.json")
    args, beam_args = parser.parse_known_args(argv)

    with open(args.schema_file, "r") as fp:
        schema_json = json.load(fp)

    pipeline_options = PipelineOptions(beam_args)
    with beam.Pipeline(options=pipeline_options) as p:
        rows = (
            p
            | "Match DICOMs" >> fileio.MatchFiles(args.input)
            | "Read Matches" >> fileio.ReadMatches()
            | "Process DICOM" >> beam.ParDo(ProcessDicomDoFn(args.output_prefix, schema_json))
        )

        rows | "WriteToBQ" >> WriteToBigQuery(
            table=args.bq_table,
            schema=_table_schema_from_json(schema_json),
            method=WriteToBigQuery.Method.STORAGE_WRITE_API,
            write_disposition=WriteToBigQuery.WriteDisposition.WRITE_APPEND,
            create_disposition=WriteToBigQuery.CreateDisposition.CREATE_IF_NEEDED,
        )


if __name__ == "__main__":
    run()
