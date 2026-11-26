"""Build nebius_plans.xlsx: two sheets, SMALL and LARGE (madry only), one row per model.

Each sheet has, top to bottom:
  Inputs    PGD steps for scratch runs (eps 1/2/4, default 5), PGD steps for continuations (eps 6/8,
            default 7), contingency on time (default 15%), number of B200s (default 16) and
            rtx_pro_6000s (8), voucher price per B200 GPU-hour.
  Summary   GPU-hours, cost and days on each GPU type, per arch and total.
  Rates     GPU-hours per epoch per (arch, PGD) on B200 and rtx_pro_6000, measured from our logs by
            measure_epoch_rates.py (epoch_rates.csv); derived cells are marked.
  Models    one row per model; its PGD and GPU-hours follow the inputs.

Usage: python build_plans.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
TODAY = date.today().isoformat()
ARCHS = ("convnext_base", "vit_b_cvst", "swin_b")
NORMS = ("linf", "l2", "l1")

INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
TOTAL_FILL = PatternFill("solid", fgColor="E2EFDA")
KEY_FILL = PatternFill("solid", fgColor="DDEBF7")
BOLD = Font(bold=True)
WHITE = Font(bold=True, color="FFFFFF")
GREY = Font(color="808080", italic=True)
H1 = Font(bold=True, size=14)
H2 = Font(bold=True, size=12)
USD = '"$"#,##0'
HRS = "#,##0"
DAYS = "0.0"
RATE = "0.000"


# ----------------------------------------------------------------------------- rates
def load_rates() -> dict:
    """(arch, pgd) -> (b200, b200_src, rtx, rtx_src); pgd 0 = clean baseline. Madry only."""
    meas = {}
    with (HERE / "epoch_rates.csv").open() as fh:
        for r in csv.DictReader(fh):
            k = (r["arch"], r["proto"], int(r["pgd"]), r["gpu"])
            if k not in meas or int(r["n_epochs"]) > int(meas[k]["n_epochs"]):
                meas[k] = r

    def m(arch, pgd, gpu):
        r = meas.get((arch, "clean" if pgd == 0 else "madry", pgd, gpu))
        return (float(r["median_gpu_h_per_epoch"]), f"measured ({r['n_models']} models)") if r else None

    ratio = m("convnext_base", 3, "B200")[0] / m("convnext_base", 3, "rtx_pro_6000")[0]
    out: dict = {"ratio": ratio}
    for arch in ARCHS:
        for pgd in (3, 5, 7, 0):
            b2, rtx = m(arch, pgd, "B200"), m(arch, pgd, "rtx_pro_6000")
            if rtx is None and arch == "vit_b_cvst":  # vit madry PGD-5/7 only ran on rtx6000
                r3, rk = m(arch, 3, "rtx6000")[0], m(arch, pgd, "rtx6000")[0]
                rtx = (m(arch, 3, "rtx_pro_6000")[0] * rk / r3, "derived (rtx6000 PGD ratio)")
            if rtx is None:
                rtx = (b2[0] / ratio, "derived (B200 / ratio)")
            if b2 is None:
                b2 = (rtx[0] * ratio, "derived (rtx_pro x ratio)")
            out[(arch, pgd)] = (b2[0], b2[1], rtx[0], rtx[1])
    return out


# ----------------------------------------------------------------------------- models
def eps_value(norm: str, k: int) -> str:
    return {"linf": f"{k}/255", "l2": f"{float(k):g}", "l1": f"{37.5 * k:g}"}[norm]


def name(arch: str, leaf: str) -> str:
    return f"convnext_base_{leaf}" if arch == "convnext_base" else f"{arch}/{leaf}"


def models(plan: str) -> list[dict]:
    sets = [("vit_b_cvst", 0), ("swin_b", 0)]
    if plan == "LARGE":
        sets = [("convnext_base", 4), ("vit_b_cvst", 0), ("vit_b_cvst", 4), ("swin_b", 0), ("swin_b", 4)]
    rows = []
    for arch, init in sets:
        for norm in NORMS:
            for k in (1, 2, 4):
                leaf = f"{norm}_{k}_init{init}"
                if arch == "convnext_base" and leaf == "l1_4_init4":
                    continue  # already trained in the lab and healthy
                rows.append(dict(arch=arch, model=name(arch, leaf), kind="scratch", eps=eps_value(norm, k), epochs=150))
        for norm in NORMS:
            for k in (6, 8):
                rows.append(dict(arch=arch, model=name(arch, f"{norm}_cont4to{k}_init{init}"), kind="continuation",
                                 eps=eps_value(norm, k), epochs=40))
        rows.append(dict(arch=arch, model=name(arch, f"baseline_init{init}"), kind="clean", eps="0", epochs=150))
    return rows


# ----------------------------------------------------------------------------- sheet
def head(ws, row: int, labels: list[str]) -> None:
    for i, lab in enumerate(labels, 1):
        c = ws.cell(row=row, column=i, value=lab)
        c.font, c.fill = WHITE, HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[row].height = 32


def build_sheet(wb: Workbook, plan: str, rates: dict) -> None:
    ws = wb.create_sheet(plan)
    rows = models(plan)
    archs = [a for a in ARCHS if any(r["arch"] == a for r in rows)]
    for col, w in zip("ABCDEFGHIJ", (40, 14, 14, 14, 14, 12, 16, 16, 14, 34)):
        ws.column_dimensions[col].width = w

    scope = {"SMALL": "ViT-B and Swin-B: 2nd madry init (init0)",
             "LARGE": "ViT-B and Swin-B: 2nd + 3rd madry init (init0, init4); ConvNeXt-B: 3rd init (init4)"}[plan]
    ws["A1"] = f"{plan} plan - {scope}"
    ws["A1"].font = H1
    ws["A2"] = ("Madry, norms L1 / L2 / Linf: eps 1, 2, 4 from scratch (150 epochs), eps 6, 8 as continuations "
                "of the eps-4 model (40 epochs), plus a clean baseline.")
    ws["A2"].font = GREY

    # ---- inputs (rows 5-10)
    ws["A4"] = "Inputs (yellow cells)"
    ws["A4"].font = H2
    inputs = [
        ("PGD steps, scratch runs (eps 1, 2, 4)", 5, "0"),
        ("PGD steps, continuation runs (eps 6, 8)", 7, "0"),
        ("Contingency added to time", 0.15, "0%"),
        ("B200 GPUs", 16, "0"),
        ("rtx_pro_6000 GPUs (our lab)", 8, "0"),
        ("Price per B200 GPU-hour (voucher PAYG)", 1.38, '"$"0.00'),
    ]
    for i, (lab, val, fmt) in enumerate(inputs):
        ws.cell(row=5 + i, column=1, value=lab)
        c = ws.cell(row=5 + i, column=2, value=val)
        c.fill, c.number_format, c.font = INPUT_FILL, fmt, BOLD
    dv = DataValidation(type="list", formula1='"3,5,7"', allow_blank=False, error="PGD steps: 3, 5 or 7")
    ws.add_data_validation(dv)
    dv.add("B5")
    dv.add("B6")
    PGD_S, PGD_C, CONT, NB200, NRTX, PRICE = "$B$5", "$B$6", "$B$7", "$B$8", "$B$9", "$B$10"

    # ---- layout rows
    st = 13                                # summary header
    tot = st + 1 + len(archs)              # summary TOTAL row
    d = tot + 2                            # time header
    rt = d + 5                             # rates header
    rt_last = rt + 4 * len(archs)
    mt = rt_last + 3                       # models header
    first, last = mt + 1, mt + len(rows)

    KEY = f"$A${rt + 1}:$A${rt_last}"
    B2COL = f"$D${rt + 1}:$D${rt_last}"
    RTXCOL = f"$F${rt + 1}:$F${rt_last}"

    # ---- rates table
    ws.cell(row=rt - 1, column=1, value="GPU-hours per epoch (measured from our training logs)").font = H2
    head(ws, rt, ["key", "arch", "PGD steps", "B200", "B200 source", "rtx_pro_6000", "rtx_pro source"])
    r = rt + 1
    for arch in archs:
        for pgd in (3, 5, 7, 0):
            b2, b2s, rtx, rtxs = rates[(arch, pgd)]
            for j, v in enumerate([f"{arch}|{pgd}", arch, pgd if pgd else "clean", b2, b2s, rtx, rtxs], 1):
                ws.cell(row=r, column=j, value=v)
            ws.cell(row=r, column=4).number_format = RATE
            ws.cell(row=r, column=6).number_format = RATE
            ws.cell(row=r, column=1).font = GREY
            for j in (5, 7):
                if str(ws.cell(row=r, column=j).value).startswith("derived"):
                    ws.cell(row=r, column=j).font = GREY
            r += 1

    # ---- model rows
    ws.cell(row=mt - 1, column=1, value=f"Models ({len(rows)})").font = H2
    head(ws, mt, ["model", "arch", "kind", "eps", "epochs", "PGD steps", "B200 GPU-h (+contingency)",
                  "rtx_pro_6000 GPU-h (+contingency)", "cost @ B200 price", "note"])
    for i, m in enumerate(rows):
        n = first + i
        for j, v in enumerate([m["model"], m["arch"], m["kind"], m["eps"], m["epochs"]], 1):
            ws.cell(row=n, column=j, value=v)
        ws.cell(row=n, column=6, value={"scratch": f"={PGD_S}", "continuation": f"={PGD_C}", "clean": 0}[m["kind"]])
        key = f'B{n}&"|"&F{n}'
        ws.cell(row=n, column=7, value=f"=E{n}*INDEX({B2COL},MATCH({key},{KEY},0))*(1+{CONT})").number_format = HRS
        ws.cell(row=n, column=8, value=f"=E{n}*INDEX({RTXCOL},MATCH({key},{KEY},0))*(1+{CONT})").number_format = HRS
        ws.cell(row=n, column=9, value=f"=G{n}*{PRICE}").number_format = USD
        note = "no attack (PGD n/a)" if m["kind"] == "clean" else ""
        if m["model"] in ("convnext_base_l2_4_init4", "convnext_base_linf_4_init4"):
            note = "retrain: lab run was gradient-masked at PGD 3"
        ws.cell(row=n, column=10, value=note).font = GREY

    # ---- summary
    ws.cell(row=st - 1, column=1, value="Summary").font = H2
    head(ws, st, ["arch", "models", "B200 GPU-h", "rtx_pro_6000 GPU-h", "cost (B200)"])
    A, G, H, I = (f"${c}${first}:${c}${last}" for c in "BGHI")
    label = {"convnext_base": "ConvNeXt-B", "vit_b_cvst": "ViT-B (CvSt)", "swin_b": "Swin-B"}
    for k, arch in enumerate(archs):
        r = st + 1 + k
        ws.cell(row=r, column=1, value=label[arch])
        ws.cell(row=r, column=2, value=f'=COUNTIF({A},"{arch}")')
        ws.cell(row=r, column=3, value=f'=SUMIF({A},"{arch}",{G})').number_format = HRS
        ws.cell(row=r, column=4, value=f'=SUMIF({A},"{arch}",{H})').number_format = HRS
        ws.cell(row=r, column=5, value=f'=SUMIF({A},"{arch}",{I})').number_format = USD
    ws.cell(row=tot, column=1, value="TOTAL")
    for j, col, fmt in ((2, "B", "0"), (3, "C", HRS), (4, "D", HRS), (5, "E", USD)):
        ws.cell(row=tot, column=j, value=f"=SUM({col}{st + 1}:{col}{tot - 1})").number_format = fmt
    for j in range(1, 6):
        ws.cell(row=tot, column=j).font = BOLD
        ws.cell(row=tot, column=j).fill = TOTAL_FILL

    # days = max(throughput, longest dependency chain: eps-4 scratch then its continuation)
    def chain(col_rates: str, per_gpu: int) -> str:
        parts = [f'(150*INDEX({col_rates},MATCH("{a}|"&{PGD_S},{KEY},0))'
                 f'+40*INDEX({col_rates},MATCH("{a}|"&{PGD_C},{KEY},0)))' for a in archs]
        return f"MAX({','.join(parts)})*(1+{CONT})*{per_gpu}/24"

    head(ws, d, ["Time to train the whole plan", "GPUs (from inputs)", "days", "days on 8 GPUs"])
    for rr, lab, n_ref, tot_col, col_rates, per_gpu in (
            (d + 1, "on B200 (2 models per GPU)", NB200, "C", B2COL, 2),
            (d + 2, "on rtx_pro_6000 (1 model per GPU)", NRTX, "D", RTXCOL, 1)):
        ws.cell(row=rr, column=1, value=lab)
        ws.cell(row=rr, column=2, value=f"={n_ref}")
        ws.cell(row=rr, column=3, value=f"=MAX({tot_col}{tot}/{n_ref}/24,{chain(col_rates, per_gpu)})")
        ws.cell(row=rr, column=4, value=f"=MAX({tot_col}{tot}/8/24,{chain(col_rates, per_gpu)})")
        for c in (3, 4):
            ws.cell(row=rr, column=c).number_format = DAYS
            ws.cell(row=rr, column=c).font = BOLD
            ws.cell(row=rr, column=c).fill = KEY_FILL
    ws.cell(row=d + 1, column=5,
            value="days = larger of: GPU-hours / GPUs / 24, or the longest eps-4 -> continuation chain").font = GREY

    # ---- notes
    nt = last + 3
    notes = [
        "GPU-hours per epoch: median of measured epochs in our logs (incl. per-epoch validation, excl. AutoAttack). "
        "B200 = AIRCC ConvNeXt runs with 2 models per GPU, so GPU-h = wall / 2; rtx_pro_6000 = Slurm, 1 model per GPU.",
        f"ViT/Swin never ran on B200: B200 = rtx_pro_6000 x {rates['ratio']:.3f} (ConvNeXt's measured "
        "B200/rtx_pro_6000 ratio). Grey sources are derived, not measured.",
        "Price: Israel Innovation Authority research voucher, B200 PAYG $1.38/GPU-h (reserve >= 3 months: $1.00). "
        f"Nebius list price B200 $8.50/GPU-h (nebius.com/prices, fetched {TODAY}).",
        "Contingency 15% covers failed / restarted runs; our measured history is 11.5% for madry runs.",
        "AutoAttack evaluation is not included (it can run on our lab GPUs).",
    ]
    if plan == "LARGE":
        notes.append("ConvNeXt init4 l1_4 is left out: already trained in the lab and healthy.")
    ws.cell(row=nt, column=1, value="Notes").font = H2
    for i, t in enumerate(notes, 1):
        ws.cell(row=nt + i, column=1, value=f"{i}. {t}").font = GREY
    ws.freeze_panes = "A4"


def build(out_dir: Path) -> Path:
    rates = load_rates()
    wb = Workbook()
    wb.remove(wb.active)
    for plan in ("SMALL", "LARGE"):
        build_sheet(wb, plan, rates)
    wb.calculation.fullCalcOnLoad = True
    out = out_dir / "nebius_plans.xlsx"
    wb.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    print(build(ap.parse_args().out_dir))


if __name__ == "__main__":
    main()
