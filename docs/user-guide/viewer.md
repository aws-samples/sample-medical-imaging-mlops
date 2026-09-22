# Prediction Viewer

!!! warning "Experimental"
    The Prediction Viewer is an experimental feature. Rendering behavior and
    supported formats may change between releases.

The Prediction Viewer is a browser-based volume viewer (built on
[NiiVue](https://github.com/niivue/niivue)) for inspecting source images and
segmentation predictions directly from S3 — no local downloads or desktop
viewer required.

!!! danger "Research Use Only"
    Rendered predictions are research outputs, not clinical diagnoses. Do not
    use the viewer as the basis for medical decisions.

## Opening the Viewer

The viewer runs as a standalone page at `/viewer`, opened in a new browser tab:

- **From the Inference tab** — click **Open Prediction Viewer**
- **From a result card** — after an inference run completes, click
  **View in viewer** (segmentation) or **View source image in viewer**
  (classification)
- **Directly by URL** — deep-link specific volumes:

```
http://localhost:8000/viewer?image=s3://bucket/imagesTs/case_1.nii.gz&mask=s3://bucket/masks/case_1_seg.nii.gz
```

## Supported Formats

| Format | Extension | Notes |
|---|---|---|
| NIfTI | `.nii`, `.nii.gz` | Streamed from S3 as-is |
| DICOM (single file) | `.dcm` | Converted server-side to NIfTI (see below) |

Both the source image and the prediction mask are optional — load either one
alone or both together as an overlay.

## How Volumes Load

Volumes are **streamed through the backend** via `GET /api/viewer/fetch`
rather than fetched from S3 directly by the browser. This means:

- **No CORS configuration is required** on your S3 buckets
- Access uses the dashboard's AWS credentials — anything the backend can
  read, the viewer can display
- Nothing is written to local disk; objects are streamed in chunks

### DICOM conversion

Browsers cannot reliably decode every DICOM transfer syntax (JPEG 2000
variants in particular). When you load a `.dcm` object, the backend decodes
it with `pydicom` + Pillow (OpenJPEG) and returns a gzipped NIfTI instead:

- Rescale slope/intercept are applied
- `MONOCHROME1` (inverted grayscale) images are normalized to standard polarity
- Pixel spacing is carried into the NIfTI affine
- 2D images (e.g., X-rays) become single-slice volumes, and the viewer
  automatically switches to axial view

## Viewer Controls

| Control | Action |
|---|---|
| **View** dropdown | Multiplanar + 3D, axial, coronal, sagittal, or 3D render |
| **Overlay opacity** slider | Fade the prediction mask over the source image |
| Drag | Change slices |
| Scroll | Zoom |
| Right-drag | Adjust contrast |

The prediction mask renders with the `nih` colormap for multi-label
segmentations.

## Automatic Prefill from Inference Runs

The Inference tab remembers your last run (per browser, via `localStorage`)
and prefills the viewer when you open it:

- **Segmentation runs** — the source image and the prediction mask are
  prefilled and loaded as an overlay
- **Classification runs** — the source image is prefilled and loaded, and the
  classification result (predicted class, confidence, and per-class
  probability bars) is shown in the right sidebar

Only S3 URI inputs can be prefilled — local file uploads have no S3 location
for the viewer to load. Explicit `?image=`/`?mask=` URL parameters always take
precedence over the remembered run.

## Recent Predictions Sidebar

The sidebar lists the most recent prediction masks written by async
endpoints. Click one to load it; add a source image URI for the overlay.

Set the bucket to scan in `.env.local`:

```bash
VIEWER_MASKS_BUCKET=your-output-bucket
```

If unset, the list is empty with a hint. The prefixes scanned are defined in
`MASK_PREFIXES` in `app/routers/viewer_router.py` — adjust them to match
where your endpoints write masks.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Load failed: ... 403` | Expired or missing AWS credentials, or no S3 read permission | Re-authenticate and restart the dashboard |
| `Load failed: ... 404` | Object doesn't exist at that URI | Check the S3 URI |
| `Could not decode DICOM (422)` | Transfer syntax not supported by the server-side decoder | Convert the file to NIfTI offline and retry |
| Viewer shows stale/garbled volume | Browser cached a previous failed response | Hard refresh (Cmd+Shift+R / Ctrl+Shift+R) |
| Recent predictions list is empty | `VIEWER_MASKS_BUCKET` unset or prefixes don't match | Set the env var and check `MASK_PREFIXES` |
