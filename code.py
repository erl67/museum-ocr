import os
import re
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output

# =========================
# CONFIG
# =========================
JPEGS_DIR = Path(r".\JPEGS")
OUT_DIR = Path(r".\out")
OCR_TEXT_DIR = OUT_DIR / "ocr_text"

WORKERS = 6
PROGRESS_EVERY = 10

# If you installed tesseract but it's not on PATH, set it explicitly:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)

ACC_RE = r"\d{1,6}"
CAT_RE = r"\d{1,7}"

# Output constants for this project
DEACCESSION_CONST = 172
DATE_CONST = "1997-05-01"

OUTPUT_COLUMNS = [
    "Page",
    "Type",
    "Accession",
    "Code",
    "Catalog",
    "Deaccession",
    "Date",
    "Temp#",
    "Genus",
    "Species",
    "PageComplete",
]

# =========================
# FILE LISTING (NO ZIP)
# =========================
def page_num_from_path(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    if not m:
        raise ValueError(f"Can't parse page number from: {p.name}")
    return int(m.group(1))

def list_jpgs_unique_by_page(folder: Path):
    files = list(folder.glob("*.[jJ][pP][gG]"))

    # Normalize + de-dupe physical paths (defensive)
    normed = []
    seen = set()
    for f in files:
        key = os.path.normcase(str(f.resolve()))
        if key not in seen:
            seen.add(key)
            normed.append(f)

    # Group by page number
    by_page = {}
    collisions = {}
    for f in normed:
        pg = page_num_from_path(f)
        by_page.setdefault(pg, []).append(f)

    # Keep largest file per page (best scan), but retain alternates
    for pg, lst in by_page.items():
        lst_sorted = sorted(lst, key=lambda x: x.stat().st_size, reverse=True)
        by_page[pg] = lst_sorted
        if len(lst_sorted) > 1:
            collisions[pg] = lst_sorted

    pages = sorted(by_page.keys())
    if len(pages) != 102 or pages[0] != 1 or pages[-1] != 102:
        raise RuntimeError(f"Expected pages 1..102. Found {len(pages)} pages: {pages[:5]} ... {pages[-5:]}")

    if collisions:
        sample = ", ".join([f"{k}:{len(v)}" for k, v in list(collisions.items())[:12]])
        print("[Info] Duplicate files for some pages (multiple files map to same page number). Will auto-try alternates if needed.")
        print(f"       Examples: {sample} ...")
    else:
        print("[Info] Found exactly 102 pages with 1 image each (no duplicates). ✅")

    tasks = []
    for pg in range(1, 103):
        paths = by_page[pg]
        tasks.append((pg, paths[0], paths[1:]))  # (page, primary, alternates)
    return tasks

# =========================
# IMAGE PREPROCESSING
# =========================
def otsu_threshold(arr_uint8: np.ndarray) -> int:
    hist = np.bincount(arr_uint8.ravel(), minlength=256).astype(np.float64)
    total = arr_uint8.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thresh = 200

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return int(thresh)

def preprocess_bw(img: Image.Image, scale=2.2, use_otsu=True, fixed_thresh=205) -> Image.Image:
    im = img.convert("L")
    im = ImageOps.autocontrast(im)
    if scale != 1.0:
        im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.Resampling.BICUBIC)
    im = im.filter(ImageFilter.MedianFilter(size=3))

    arr = np.array(im)
    thr = otsu_threshold(arr) if use_otsu else fixed_thresh
    bw = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(bw)

# =========================
# OCR HELPERS
# =========================
def ocr_data(img_bw: Image.Image, psm: int) -> dict:
    whitelist = "0123456789EDG"
    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_data(img_bw, config=cfg, output_type=Output.DICT)

def ocr_text_for_debug(img_bw: Image.Image, psm: int) -> str:
    whitelist = "0123456789EDG"
    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img_bw, config=cfg)

def ocr_find_headers(img: Image.Image) -> dict:
    """
    Header detection on the ORIGINAL (unscaled) image.
    Returns y-coordinates in ORIGINAL image coords.
    """
    im = img.convert("L")
    im = ImageOps.autocontrast(im)

    cfg = "--oem 3 --psm 6"
    d = pytesseract.image_to_data(im, config=cfg, output_type=Output.DICT)

    headers = {"PICKLES": [], "SKELETONS": [], "SKINS": []}
    for txt, top, h in zip(d["text"], d["top"], d["height"]):
        if not txt:
            continue
        t = re.sub(r"[^A-Z]", "", txt.upper())
        if t in headers:
            headers[t].append(int(top) + int(h) // 2)  # center y (original coords)
    return headers

def clean_token(t: str) -> str:
    t = (t or "").strip().upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t

def split_subtokens(text: str):
    if not text:
        return []
    m = re.fullmatch(rf"({ACC_RE})([EDG])({CAT_RE})", text)
    if m:
        return [m.group(1), m.group(2), m.group(3)]
    m = re.fullmatch(rf"({ACC_RE})([EDG])", text)
    if m:
        return [m.group(1), m.group(2)]
    m = re.fullmatch(rf"([EDG])({CAT_RE})", text)
    if m:
        return [m.group(1), m.group(2)]
    return [text]

def cluster_rows(tokens, row_tol):
    tokens = sorted(tokens, key=lambda d: d["y"])
    rows = []
    cur = []
    cur_y = None
    for tok in tokens:
        if cur_y is None:
            cur = [tok]
            cur_y = tok["y"]
            continue
        if abs(tok["y"] - cur_y) <= row_tol:
            cur.append(tok)
            cur_y = (cur_y * (len(cur) - 1) + tok["y"]) / len(cur)
        else:
            rows.append(cur)
            cur = [tok]
            cur_y = tok["y"]
    if cur:
        rows.append(cur)
    return rows

def kmeans_1d(xs, k=3, iters=30):
    xs = np.array(xs, dtype=float)
    centers = np.percentile(xs, np.linspace(0, 100, k + 2)[1:-1])
    for _ in range(iters):
        d = np.abs(xs[:, None] - centers[None, :])
        lab = d.argmin(axis=1)
        new = []
        for j in range(k):
            pts = xs[lab == j]
            new.append(centers[j] if len(pts) == 0 else pts.mean())
        new = np.array(new)
        if np.allclose(new, centers):
            break
        centers = new
    return np.sort(centers)

# =========================
# TRIPLET EXTRACTION (COLUMN 1, THEN COLUMN 2, THEN COLUMN 3)
# =========================
def extract_triplets(img_bw: Image.Image, psm=6, conf_floor=-1.0, kcols=3, return_meta=False):
    """
    Extracts triplets in COLUMN-MAJOR order:
      - all of column 1 top->bottom,
      - then all of column 2 top->bottom,
      - then all of column 3 top->bottom.

    No sorting by accession.
    """
    w, h = img_bw.size
    data = ocr_data(img_bw, psm=psm)

    tokens = []
    xs = []
    for raw, left, top, width, height, conf in zip(
        data["text"], data["left"], data["top"], data["width"], data["height"], data["conf"]
    ):
        if not raw or raw.strip() == "":
            continue
        try:
            conf = float(conf)
        except:
            conf = -1.0
        if conf < conf_floor:
            continue

        t = clean_token(raw)
        if not t:
            continue

        x = int(left) + int(width) / 2.0
        y = int(top) + int(height) / 2.0
        tokens.append({"text": t, "x": x, "y": y, "conf": conf})
        xs.append(x)

    if not tokens:
        return ([], []) if return_meta else []

    centers = kmeans_1d(xs, k=kcols)  # sorted => col index corresponds left->right
    row_tol = max(12.0, h * 0.007)
    rows = cluster_rows(tokens, row_tol)

    trip_items = []  # list of dicts: acc,code,cat,y,col,x
    seen = set()

    # For second pass inference we need row/col streams with y/x
    row_col_streams = []  # list[row][col] = list[subtoken dicts sorted by x]

    for row in rows:
        subs = []
        for tok in row:
            col = int(np.argmin(np.abs(centers - tok["x"])))
            for s in split_subtokens(tok["text"]):
                subs.append({"text": s, "x": tok["x"], "y": tok["y"], "conf": tok["conf"], "col": col})

        col_streams = []
        for col in range(kcols):
            cs = sorted([s for s in subs if s["col"] == col], key=lambda d: d["x"])
            col_streams.append(cs)
        row_col_streams.append(col_streams)

        # strict triplets
        for col in range(kcols):
            cs = col_streams[col]
            j = 0
            while j <= len(cs) - 3:
                a, c, b = cs[j]["text"], cs[j + 1]["text"], cs[j + 2]["text"]
                if re.fullmatch(ACC_RE, a) and c in ("E", "D", "G") and re.fullmatch(CAT_RE, b):
                    tup = (int(a), c, int(b))
                    if tup not in seen:
                        ytrip = (cs[j]["y"] + cs[j + 1]["y"] + cs[j + 2]["y"]) / 3.0
                        xtrip = (cs[j]["x"] + cs[j + 1]["x"] + cs[j + 2]["x"]) / 3.0
                        trip_items.append(
                            {"acc": tup[0], "code": tup[1], "cat": tup[2], "y": float(ytrip), "col": int(col), "x": float(xtrip)}
                        )
                        seen.add(tup)
                    j += 3
                else:
                    j += 1

    if not trip_items:
        return ([], []) if return_meta else []

    # infer dominant code/accession from strict pass
    codes = [d["code"] for d in trip_items]
    accs = [d["acc"] for d in trip_items]
    mode_code, code_ct = Counter(codes).most_common(1)[0]
    mode_acc, acc_ct = Counter(accs).most_common(1)[0]
    code_share = code_ct / max(1, len(codes))
    acc_share = acc_ct / max(1, len(accs))

    # second pass: allow "ACC CATALOG" (missing code)
    if code_share >= 0.85:
        for col_streams in row_col_streams:
            for col in range(kcols):
                cs = col_streams[col]
                for j in range(len(cs) - 1):
                    a, b = cs[j]["text"], cs[j + 1]["text"]
                    if re.fullmatch(ACC_RE, a) and re.fullmatch(CAT_RE, b):
                        tup = (int(a), mode_code, int(b))
                        if tup not in seen:
                            ytrip = (cs[j]["y"] + cs[j + 1]["y"]) / 2.0
                            xtrip = (cs[j]["x"] + cs[j + 1]["x"]) / 2.0
                            trip_items.append(
                                {"acc": tup[0], "code": tup[1], "cat": tup[2], "y": float(ytrip), "col": int(col), "x": float(xtrip)}
                            )
                            seen.add(tup)

    # repair accession like 2542 -> 254 when 254 dominates the page
    if acc_share >= 0.50:
        mstr = str(mode_acc)
        for it in trip_items:
            a = it["acc"]
            astr = str(a)
            if a != mode_acc and astr.startswith(mstr) and len(astr) == len(mstr) + 1:
                cand = (mode_acc, it["code"], it["cat"])
                old = (it["acc"], it["code"], it["cat"])
                if cand not in seen:
                    if old in seen:
                        seen.remove(old)
                    it["acc"] = mode_acc
                    seen.add(cand)

    # FINAL ORDER: COLUMN-MAJOR (col 0 then 1 then 2), each top-to-bottom
    trip_items.sort(key=lambda d: (d["col"], d["y"], d["x"]))

    triplets = [(d["acc"], d["code"], d["cat"]) for d in trip_items]

    if not return_meta:
        return triplets

    meta = [{"y": d["y"], "col": d["col"], "x": d["x"]} for d in trip_items]
    return triplets, meta

# =========================
# TYPE / EXPECTED COUNTS
# =========================
def base_type_for_page(page: int) -> str:
    if page <= 99:
        return "SKINS"
    if page == 100:
        return "PICKLES"
    return "SKELETONS"

def expected_rows_for_page(page: int, headers_found: dict) -> int:
    if page == 1:
        return 141
    if page == 102:
        return 136

    exp = 162
    header_present = (len(headers_found.get("PICKLES", [])) > 0) or (len(headers_found.get("SKELETONS", [])) > 0)
    if header_present:
        exp -= 2
    return exp

# =========================
# PAGE PROCESSING
# =========================
def process_one(task):
    page, primary_path, alternates = task

    img = Image.open(primary_path)
    headers = ocr_find_headers(img)
    expected = expected_rows_for_page(page, headers)

    attempts = [
        # (scale, use_otsu, fixed_thresh, psm)
        (2.2, True, 205, 6),
        (2.6, True, 205, 6),
        (2.2, False, 200, 6),
        (2.6, False, 200, 6),
        (2.2, True, 205, 4),
        (2.6, True, 205, 4),
    ]

    tried_paths = [primary_path] + list(alternates)
    best = None

    for path_try in tried_paths:
        img_try = Image.open(path_try)
        for (scale, use_otsu, thr, psm) in attempts:
            bw = preprocess_bw(img_try, scale=scale, use_otsu=use_otsu, fixed_thresh=thr)

            need_meta = (page in (99, 100))
            if need_meta:
                triplets, meta = extract_triplets(bw, psm=psm, conf_floor=-1.0, kcols=3, return_meta=True)
            else:
                triplets = extract_triplets(bw, psm=psm, conf_floor=-1.0, kcols=3)
                meta = []

            debug_path = OCR_TEXT_DIR / f"page_{page:03d}.txt"
            debug_text = ocr_text_for_debug(bw, psm=psm)
            debug_path.write_text(debug_text, encoding="utf-8", errors="ignore")

            if best is None or len(triplets) > len(best["triplets"]):
                best = {
                    "triplets": triplets,
                    "meta": meta,
                    "path": path_try,
                    "debug": debug_path,
                    "psm": psm,
                    "scale": scale,
                }

            if len(triplets) == expected:
                return {
                    "page": page,
                    "rows": triplets,
                    "rows_meta": meta,       # aligned with rows
                    "rows_extracted": len(triplets),
                    "rows_expected": expected,
                    "error": "",
                    "attempts_used": 1,
                    "image_file": str(path_try),
                    "debug_file": str(debug_path),
                    "headers": headers,      # original-image coords
                    "scale_used": scale,     # bw coords = original * scale_used
                }

    return {
        "page": page,
        "rows": best["triplets"] if best else [],
        "rows_meta": best["meta"] if best else [],
        "rows_extracted": len(best["triplets"]) if best else 0,
        "rows_expected": expected,
        "error": f"COUNT_MISMATCH {len(best['triplets']) if best else 0} != {expected}",
        "attempts_used": len(tried_paths) * len(attempts),
        "image_file": str(best["path"]) if best else str(primary_path),
        "debug_file": str(best["debug"]) if best else "",
        "headers": headers,
        "scale_used": best["scale"] if best else 1.0,
    }

# =========================
# RUN ALL + OUTPUTS
# =========================
def run_all():
    TARGET_TOTAL_ROWS = 16473  # your target end result

    tasks = list_jpgs_unique_by_page(JPEGS_DIR)

    print(f"[Start] Processing {len(tasks)} pages from {JPEGS_DIR} with {WORKERS} workers")
    t0 = time.time()

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1

            if done % PROGRESS_EVERY == 0 or done == len(tasks):
                rate = done / max(1e-9, (time.time() - t0))
                remaining = len(tasks) - done
                eta = remaining / max(1e-9, rate)
                print(f"  completed {done}/{len(tasks)} | {rate:.2f} pages/sec | ETA ~ {eta/60:.1f} min")

    results_sorted = sorted(results, key=lambda r: r["page"])

    page_counts = []
    needs_review = []

    all_rows = []
    complete_rows = []

    for r in results_sorted:
        page = r["page"]
        extracted = int(r["rows_extracted"])
        expected = int(r["rows_expected"])
        err = (r["error"] or "").strip()
        is_complete = (err == "")

        pct = (extracted / expected * 100.0) if expected else 0.0

        rec = {
            "Page": page,
            "RowsExtracted": extracted,
            "RowsExpected": expected,
            "PctComplete": round(pct, 2),
            "Complete": bool(is_complete),
            "Error": err,
            "AttemptsUsed": int(r["attempts_used"]),
            "ImageFile": r["image_file"],
            "OcrTextFile": r["debug_file"],
        }
        page_counts.append(rec)

        if err:
            needs_review.append(rec)

        base_type = base_type_for_page(page)

        rows = r.get("rows") or []
        metas = r.get("rows_meta") or []
        if len(metas) != len(rows):
            metas = [{} for _ in rows]

        headers = r.get("headers") or {"PICKLES": [], "SKELETONS": [], "SKINS": []}
        scale_used = float(r.get("scale_used") or 1.0)

        # Page 99 pivot: only affects column 3 (col=2)
        pivot99 = None
        if page == 99:
            if headers.get("PICKLES"):
                pivot99 = float(min(headers["PICKLES"])) * scale_used
            else:
                ys = sorted([m.get("y") for m in metas if m.get("col") == 2 and m.get("y") is not None])
                # PICKLES starts after the 8th line down in column 3 => pivot between 8th and 9th
                if len(ys) >= 9:
                    pivot99 = (ys[7] + ys[8]) / 2.0
                elif len(ys) >= 8:
                    pivot99 = ys[7] + 1.0

        # Page 100 pivot: ONLY column 1 has PICKLES; columns 2/3 are SKELETONS throughout
        pivot100 = None
        if page == 100:
            if headers.get("SKELETONS"):
                pivot100 = float(min(headers["SKELETONS"])) * scale_used
            else:
                ys0 = sorted([m.get("y") for m in metas if m.get("col") == 0 and m.get("y") is not None])
                # SKELETONS starts in column 1 after 27 PICKLES lines => pivot between 27th and 28th
                if len(ys0) >= 28:
                    pivot100 = (ys0[26] + ys0[27]) / 2.0
                elif len(ys0) >= 27:
                    pivot100 = ys0[26] + 1.0

        def type_for_row(p, col, y):
            if p == 99:
                # Columns 1–2 always SKINS; only col 3 flips
                if col in (0, 1) or col is None:
                    return "SKINS"
                if col == 2 and (pivot99 is not None) and (y is not None):
                    return "PICKLES" if y > pivot99 else "SKINS"
                return "SKINS"

            if p == 100:
                # FIX: PICKLES only exists in column 1 (col=0) for first 27 rows; everything else is SKELETONS.
                if col in (1, 2):
                    return "SKELETONS"
                if col == 0 and (pivot100 is not None) and (y is not None):
                    return "SKELETONS" if y > pivot100 else "PICKLES"
                # fallback defaults
                return "PICKLES" if col == 0 else "SKELETONS"

            return base_type

        for (trip, m) in zip(rows, metas):
            acc, code, cat = trip
            col = m.get("col")
            y = m.get("y")

            row = {
                "Page": page,
                "Type": type_for_row(page, col, y),
                "Accession": acc,
                "Code": code,
                "Catalog": cat,
                "Deaccession": DEACCESSION_CONST,
                "Date": DATE_CONST,
                "Temp#": "",
                "Genus": "",
                "Species": "",
                "PageComplete": bool(is_complete),
            }
            all_rows.append(row)
            if is_complete:
                complete_rows.append(row)

    page_counts_df = pd.DataFrame(page_counts)
    needs_review_df = pd.DataFrame(needs_review)
    all_df = pd.DataFrame(all_rows)
    complete_df = pd.DataFrame(complete_rows)

    # Enforce column order exactly
    if not all_df.empty:
        all_df = all_df.reindex(columns=OUTPUT_COLUMNS)
    if not complete_df.empty:
        complete_df = complete_df.reindex(columns=OUTPUT_COLUMNS)

    out_csv = OUT_DIR / "output.csv"
    out_complete_csv = OUT_DIR / "output_complete_pages.csv"
    out_counts_csv = OUT_DIR / "page_counts.csv"
    out_review_csv = OUT_DIR / "needs_review.csv"
    out_summary_txt = OUT_DIR / "run_summary.txt"

    all_df.to_csv(out_csv, index=False)
    complete_df.to_csv(out_complete_csv, index=False)
    page_counts_df.to_csv(out_counts_csv, index=False)
    needs_review_df.to_csv(out_review_csv, index=False)

    extracted_total = int(len(all_df))
    missing_total = int(TARGET_TOTAL_ROWS - extracted_total)
    pct_total = (extracted_total / TARGET_TOTAL_ROWS * 100.0) if TARGET_TOTAL_ROWS else 0.0

    expected_total_rule = int(page_counts_df["RowsExpected"].sum()) if not page_counts_df.empty else 0
    missing_vs_rule = int(expected_total_rule - extracted_total)
    pct_vs_rule = (extracted_total / expected_total_rule * 100.0) if expected_total_rule else 0.0

    complete_pages = int(page_counts_df["Complete"].sum()) if not page_counts_df.empty else 0
    total_pages = int(len(page_counts_df))

    summary_lines = [
        "=== OCR RUN SUMMARY ===",
        "",
        f"Pages processed:            {total_pages}",
        f"Pages complete:             {complete_pages}/{total_pages} ({(complete_pages/total_pages*100.0 if total_pages else 0.0):.2f}%)",
        f"Pages flagged:              {total_pages - complete_pages}",
        "",
        "=== ROW COUNTS ===",
        f"Target rows (your target):  {TARGET_TOTAL_ROWS}",
        f"Extracted rows (actual):    {extracted_total}",
        f"Missing vs your target:     {missing_total}",
        f"Completion vs your target:  {pct_total:.2f}%",
        "",
        f"Rule-based expected total:  {expected_total_rule}",
        f"Missing vs rule-based:      {missing_vs_rule}",
        f"Completion vs rule-based:   {pct_vs_rule:.2f}%",
        "",
        "=== OUTPUT FILES ===",
        f"All rows (includes failed): {out_csv}",
        f"Complete pages only:        {out_complete_csv}",
        f"Page counts:                {out_counts_csv}",
        f"Needs review:               {out_review_csv}",
        f"Debug dumps:                {OCR_TEXT_DIR}\\page_###.txt",
        "",
        "=== CSV COLUMNS (ORDER) ===",
        "\t".join(OUTPUT_COLUMNS),
    ]
    out_summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n[Done] Outputs:")
    print(f"  - {out_csv}                 rows={len(all_df)} (includes incomplete pages; see PageComplete)")
    print(f"  - {out_complete_csv}        rows={len(complete_df)} (only pages that passed checks)")
    print(f"  - {out_counts_csv}")
    print(f"  - {out_review_csv}          pages_flagged={len(needs_review_df)}")
    print(f"  - {out_summary_txt}")
    print(f"\n[Summary] Target={TARGET_TOTAL_ROWS} | Extracted={extracted_total} | Missing={missing_total} | {pct_total:.2f}%")

    return page_counts_df, needs_review_df, complete_df, all_df

# Run:
page_counts_df, needs_review_df, complete_df, all_df = run_all()