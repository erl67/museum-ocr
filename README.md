# Museum OCR — CMNH Bird Collection Deaccessions

**Optical character recognition and data extraction from historical Carnegie Museum of Natural History bird collection deaccession records.**

This project digitizes a 102-page collection of printed deaccession records from the bird collection at the Carnegie Museum of Natural History (CMNH) in Pittsburgh, Pennsylvania. The original documents, dating to the 1997–1999 deaccession period, were scanned and processed using Python and Tesseract OCR to convert thousands of printed entries into structured, searchable data.

The project involved automated extraction, image preprocessing, validation against known record counts, targeted rescanning of problematic pages, and manual verification against the original documents.

## Results

| Metric | Final result |
|---|---:|
| Pages processed | 102 |
| Expected records | 16,473 |
| Records extracted | 16,473 |
| Row-count completion | **100.00%** |
| Pages passing count validation | 102 / 102 |
| Pages requiring review in final OCR run | 0 |

The initial best OCR result contained **16,454 of 16,473 records (99.88%)**, leaving 19 missing entries across five problematic pages.

After rescanning those pages, the OCR system successfully extracted all 16,473 expected records. The results were subsequently checked against the scanned originals, and remaining transcription errors were corrected manually.

**Important:** The 100.00% figure represents complete record-count extraction, not independently measured character-level OCR accuracy. Reaching the expected number of records did not eliminate the need for manual verification.

## Repository Contents

| File / Directory | Description |
|---|---|
| [`DeaccessionScans.pdf`](DeaccessionScans.pdf) | Original scanned document, containing the 102-page deaccession packet followed by additional greensheets |
| [`JPEGS/`](JPEGS/) | Individual JPEG scans numbered 1–102, used as OCR inputs |
| [`code.py`](code.py) | Python OCR processing program |
| [`ocr.ipynb`](ocr.ipynb) | Jupyter notebook for running the OCR workflow |
| [`out/`](out/) | Extracted CSV data, processing summaries, and OCR diagnostic files |
| [`deaccessions.xlsx`](deaccessions.xlsx) | Consolidated deaccession spreadsheet containing the project data and additional sections |
| [`out/green_pages.ods`](out/green_pages.ods) | Supplemental greensheet data |
| [`out/g1_g4_1999-05-01_skins.csv`](out/g1_g4_1999-05-01_skins.csv) | Supplemental skins records |

The primary OCR workflow processes the 102 numbered JPEG pages. The additional greensheets are separate source material and are not included in the 16,473-record OCR target.

## How It Works

The program uses **Python, Tesseract OCR, Pillow, NumPy, and pandas** to recognize and extract structured records from scanned pages.

Each page contains three vertical columns of entries, with each entry comprising three fields:

- **Accession:** Collection accession number
- **Code:** Letter identifier (`E`, `D`, or `G`)
- **Catalog:** Catalog number

The processing workflow consists of the following stages.

### 1. Image Loading and Preprocessing

The program loads the 102 JPEG images and prepares them for OCR.

Preprocessing includes grayscale conversion, automatic contrast adjustment, image enlargement using bicubic interpolation, median filtering to reduce noise, and binary thresholding.

Both automatic Otsu thresholding and fixed thresholds are supported.

The program can attempt multiple preprocessing configurations when a page does not produce the expected record count.

### 2. OCR Recognition

Tesseract recognizes individual text elements and provides their contents, confidence scores, and positional coordinates.

Recognition is restricted to the characters expected in the source documents: digits `0–9` and letters `E`, `D`, and `G`.

The program uses Tesseract's positional data rather than relying exclusively on plain-text OCR output. This is necessary because the source documents contain multiple columns of closely spaced numerical records.

### 3. Row and Column Reconstruction

Recognized text elements are grouped using their coordinates.

The program uses vertical positions to reconstruct printed rows and one-dimensional k-means clustering to identify the three horizontal column groups.

Within each row and column, tokens are interpreted as accession/code/catalog triplets.

Additional parsing logic handles cases where Tesseract combines adjacent fields into a single token, or where a missing code can be inferred from surrounding entries.

The program also includes limited accession-number repair logic for certain recognized digit-insertion patterns.

### 4. Preserving Original Document Order

The final output preserves the physical reading order of the original pages:

1. Column 1, top to bottom
2. Column 2, top to bottom
3. Column 3, top to bottom
4. Continue to the next page

Records are **not sorted by accession or catalog number**.

Preserving this order makes it considerably easier to compare extracted data against the original scans during review.

### 5. Section Classification

The original deaccession records contain three specimen categories:

- `SKINS`
- `PICKLES` — fluid-preserved specimens
- `SKELETONS`

Most pages contain a single category, but pages 99 and 100 contain transitions between sections.

These require special handling to ensure that each record receives the correct `Type` value.

**Page 99:** The first two columns contain skins. The third column begins with eight skins entries, followed by the `PICKLES` heading. All subsequent entries in that column are classified as pickles.

**Page 100:** Only the first 27 entries of the first column are pickles. The `SKELETONS` section begins after those entries and continues through the remainder of column 1 and all of columns 2 and 3.

The program uses OCR positional metadata and section headings, with fallback logic based on the known transition positions.

### 6. Validation and Retry Processing

A key feature of the project is its use of known record counts to evaluate extraction completeness.

Most pages contain 162 records, with several exceptions:

| Pages | Expected records per page |
|---|---:|
| 1 | 141 |
| 2–98 | 162 |
| 99 | 160 |
| 100 | 160 |
| 101 | 162 |
| 102 | 136 |
| **Total** | **16,473** |

Pages that fail to meet their expected count can be reprocessed using alternative image scales, thresholds, and Tesseract page segmentation modes.

The program uses six parallel workers by default and reports progress, processing speed, and estimated remaining time.

At completion, it generates an extraction summary identifying the total target, completed records, missing records, percentage completion, and pages requiring further review.

## Challenges and Solutions

### Missing Records and Problematic Scans

The initial best result extracted 16,454 records, leaving 19 missing across five pages:

| Page | Extracted | Expected | Missing |
|---|---:|---:|---:|
| 20 | 158 | 162 | 4 |
| 27 | 160 | 162 | 2 |
| 53 | 161 | 162 | 1 |
| 61 | 156 | 162 | 6 |
| 62 | 156 | 162 | 6 |
| **Total** | **791** | **810** | **19** |

Pages 61 and 62 were particularly problematic because of noticeable scan skew.

Several software-based improvements were tested, including automatic deskewing, separate-column OCR, and revised row-clustering logic. These changes did not improve the overall result and, in some configurations, caused substantial reductions in extracted records.

The original processing approach was retained.

**Resolution:** The five problematic pages were rescanned. The improved source images allowed the existing OCR implementation to extract all expected records without requiring major algorithmic changes.

### Accession Number Recognition Errors

Although rescanning resolved the missing-record problem, a number of incorrect values remained in the accession field.

These included:

- Digit substitutions, particularly confusion between `5` and `9`
- Unexpected digit insertions, such as `1234` becoming `19234`
- Repeated digit sequences, such as `7234` becoming `723434` or `727234`

The errors tended to cluster on particular pages, suggesting scan quality, printing irregularities, and character segmentation as contributing factors.

Some recognition errors were difficult to detect automatically because the resulting strings still consisted of valid numerical characters.

**Resolution:** The extracted records were reviewed against the original scanned pages, and incorrect accession values were manually corrected.

This demonstrated an important limitation of count-based validation: **a page can contain the correct number of extracted records while still containing transcription errors.**

### Multi-Column Reading Order

Initial output was organized by accession number rather than by physical location on the printed page.

A subsequent revision produced row-major ordering, moving from the left column to the right column on each horizontal line. Although spatially consistent, this did not match how the original document was intended to be read.

**Resolution:** Positional metadata was retained for extracted records, and final output was sorted by column followed by vertical position.

### Section Boundaries on Pages 99 and 100

Initial output assigned specimen types at the page level, causing records near the `PICKLES` and `SKELETONS` headings to receive incorrect classifications.

A simple page-wide vertical transition was also insufficient because the printed sections changed partway through individual columns.

**Resolution:** Column-aware classification rules were introduced for pages 99 and 100, matching the actual transitions visible in the source documents.

### Verification and Final Reconciliation

The final stage involved comparing the extracted records with the original scans, identifying remaining errors, and reconciling them manually.

This review was especially important for accession numbers, where incorrect values could otherwise remain undetected despite a successful page-count validation.

The completed project therefore combines automated OCR with human review, rather than relying solely on the program's reported completion percentage.

## Output Data

The primary output is [`out/output.csv`](out/output.csv).

Its columns are arranged as follows:

| Column | Description |
|---|---|
| `Page` | Original scanned page number |
| `Type` | Specimen category: SKINS, PICKLES, or SKELETONS |
| `Accession` | Museum accession number |
| `Code` | Original record code: E, D, or G |
| `Catalog` | Catalog number |
| `Deaccession` | Constant value `172` |
| `Date` | Constant value `1997-05-01` |
| `Temp#` | Blank placeholder |
| `Genus` | Blank placeholder |
| `Species` | Blank placeholder |
| `PageComplete` | Boolean indicating whether the page passed row-count validation |

The constant fields and blank placeholders maintain compatibility with the project's intended data structure. They are not independently extracted from each printed entry.

### Generated Files

| Output | Purpose |
|---|---|
| `out/output.csv` | All extracted records, including records from incomplete pages |
| `out/output_complete_pages.csv` | Records from pages that pass count validation |
| `out/page_counts.csv` | Per-page extracted and expected counts, completion percentages, and processing details |
| `out/needs_review.csv` | Pages that fail count validation |
| `out/run_summary.txt` | Overall processing results and completion statistics |
| `out/ocr_text/page_###.txt` | Diagnostic OCR text for individual pages |

During processing, records from incomplete pages are retained in the main output rather than discarded. The `PageComplete` field indicates whether the corresponding page passed its count check.

In the final published OCR run, all 102 pages passed, so both main output files contain 16,473 records.

The OCR text dumps are diagnostic artifacts; they are not substitutes for the structured output or the original page scans.

## Requirements and Usage

The program was developed for Python on Windows using a locally installed copy of Tesseract OCR.

### Dependencies

Python packages:

- `numpy`
- `pandas`
- `Pillow`
- `pytesseract`

External dependency:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

Install the required Python packages:

```bash
pip install numpy pandas pillow pytesseract
```

Install Tesseract separately and confirm that its executable path matches the configuration in `code.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Running the Program

Clone the repository:

```bash
git clone https://github.com/erl67/museum-ocr.git
cd museum-ocr
```

Ensure that the `JPEGS` directory contains the 102 numbered input files (`1.jpg` through `102.jpg`).

Run the Python script from the repository root:

```bash
python code.py
```

Alternatively, the project includes `ocr.ipynb` for use in Jupyter.

The program processes the images using six workers by default. This can be adjusted through the `WORKERS` configuration variable.

Processing progress is reported periodically, and the generated files are written to the `out` directory.

The program's current validation rules are specific to the layout and expected record counts of this document set. Applying it to different collections or page layouts would require modifications.

**Reproducibility note:** Manual accession corrections are not necessarily encoded as OCR processing rules. Regenerating the CSV files may reproduce the automated extraction counts without reproducing every subsequent human correction. Newly generated outputs should therefore be checked against the reviewed data before replacing the final records.

## Final Outcome

The project successfully converted the complete 102-page deaccession packet into **16,473 structured records**, with all pages accounted for and remaining detected transcription errors reconciled against the original source material.

The resulting data can be searched, sorted, filtered, cross-referenced, and incorporated into other museum collection-management or research workflows.

The project also illustrates several practical lessons in document digitization:

- Correcting the source scans can be more effective than increasing OCR processing complexity.
- Known page counts provide an effective way to detect missing records, but do not establish transcription accuracy.
- Physical page layout matters when extracting records from historical multi-column documents.
- Automated extraction combined with targeted human verification is an effective approach for converting large historical datasets into usable digital records.

## Project Status

**Complete — November 2026**

All 102 pages processed, all 16,473 expected records extracted, and identified transcription errors reconciled following manual review.

## Disclaimer

This is an independent volunteer project and is not an official publication of Carnegie Museums of Pittsburgh or Carnegie Museum of Natural History. The project is not sponsored or endorsed by either institution.
