"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   your algorithm / model (the research part)
    03 rank      prioritise for a fixed review budget B
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK (or track equivalent) with reason_code + owner, write outputs

Required output: out/<pred_file> with the columns listed for your track in
docs/tracks/TXX.md (also in src/metrics/tracks.py::TRACKS). Then evaluate with
    python -m src.evaluate --track tXX --pred out/<pred_file>

Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
See src/examples/random_baseline.py for a minimal end-to-end wiring example.
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.image_ops import brightness_stats, laplacian_variance, load_rgb
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images_path = ctx.input_dir / "images.csv"
    if not images_path.exists():
        raise FileNotFoundError(f"T02 input table not found: {images_path}")

    images = pd.read_csv(images_path)
    required_columns = {"image_id", "path", "camera_id", "light", "split"}
    missing_columns = sorted(required_columns - set(images.columns))
    if missing_columns:
        raise ValueError(f"images.csv is missing required columns: {missing_columns}")

    data = {
        "manifest": pd.read_csv(manifest) if manifest.exists() else None,
        "images": images,
    }
    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    if ctx.track != "t02":
        spec = get_track(ctx.track)
        raise NotImplementedError(f"This group pipeline currently implements t02 only, not {spec.track}")

    rows: list[dict] = []
    for item in data["images"].itertuples(index=False):
        image_path = ctx.input_dir / str(item.path)
        if not image_path.exists():
            raise FileNotFoundError(f"image referenced by images.csv not found: {image_path}")
        image = load_rgb(image_path)
        stats = brightness_stats(image)
        rows.append(
            {
                "image_id": str(item.image_id),
                "path": str(item.path),
                "camera_id": str(item.camera_id),
                "light": str(item.light),
                "split": str(item.split),
                "sharpness": laplacian_variance(image),
                "brightness": stats["mean"],
                "frac_dark": stats["frac_dark"],
                "frac_saturated": stats["frac_saturated"],
            }
        )

    scored = pd.DataFrame(rows)
    calibration = scored[scored["split"].eq("dev")]
    if calibration.empty:
        calibration = scored

    # Calibrate on the unlabeled dev distribution only; no ground truth is read.
    sharp_q10 = float(calibration["sharpness"].quantile(0.10))
    sharp_q50 = float(calibration["sharpness"].quantile(0.50))
    sharp_scale = max(sharp_q50 - sharp_q10, 1e-6)
    scored["blur_risk"] = ((sharp_q50 - scored["sharpness"]) / sharp_scale).clip(0.0, 1.0)
    scored["dark_risk"] = ((45.0 - scored["brightness"]) / 45.0).clip(0.0, 1.0)
    scored["glare_risk"] = ((scored["frac_saturated"] - 0.01) / 0.15).clip(0.0, 1.0)
    scored["score"] = (
        0.55 * scored["blur_risk"]
        + 0.25 * scored["dark_risk"]
        + 0.20 * scored["glare_risk"]
    ).clip(0.0, 1.0)
    scored["reason"] = scored.apply(
        lambda r: ",".join(
            name
            for name, value in (
                ("blur", r["blur_risk"]),
                ("dark", r["dark_risk"]),
                ("glare", r["glare_risk"]),
            )
            if value >= 0.35
        )
        or "low_quality_signal",
        axis=1,
    )
    return scored.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Default: top-B by `score`. Adapt id/score/reason columns to your track."""
    budget = int(ctx.param("budget", 20))
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column (higher = review first), "
                         "or adapt rank() to your track")
    q = make_review_queue(scored, id_col=scored.columns[0], score_col="score", budget=budget,
                          reason_col="reason" if "reason" in scored.columns else None,
                          evidence_col="path" if "path" in scored.columns else None)
    write_review_queue(q, ctx.out("review_queue.csv"))
    if "path" in scored.columns:
        render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                             title=f"{ctx.track} raw QC review queue")
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """If a reviewed queue exists (out/review_queue_reviewed.csv), apply it. Otherwise pass through.

    TODO: decide what an `edit` / `reject` means for your track and apply it
    (e.g. via common.review.apply_overrides), logging every change to ctx.audit.
    """
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        reviewed = load_review_decisions(reviewed_path)
        ctx.stats["n_reviewed"] = len(reviewed)
        for row in reviewed.itertuples(index=False):
            ctx.audit.log(
                actor=row.reviewer,
                action=f"review:{row.reviewer_decision}",
                item_id=row.item_id,
                reason=row.review_note,
            )
        if len(reviewed):
            edited = reviewed[reviewed["override_value"].astype(str).str.strip() != ""].copy()
            if len(edited):
                edited["override_value"] = pd.to_numeric(edited["override_value"], errors="raise")
                scored = apply_overrides(scored, edited, id_col="image_id", value_col="score", audit=ctx.audit)
                scored["score"] = pd.to_numeric(scored["score"], errors="raise").clip(0.0, 1.0)
    return scored


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    queue_size = min(int(ctx.param("budget", 20)), len(final))
    top_score = float(final["score"].max()) if len(final) else 0.0
    status = "HOLD" if queue_size and top_score >= 0.35 else "GO"
    reason_code = "QC_REVIEW_REQUIRED" if status == "HOLD" else "NO_HIGH_RISK_SIGNAL"
    reason = (
        f"Top-{queue_size} raw-QC candidates require human review (max score={top_score:.3f})."
        if status == "HOLD"
        else "No high-risk raw-QC signal exceeded the frozen review threshold."
    )
    return Decision(
        status=status,
        reason_code=reason_code,
        reason=reason,
        owner=ctx.config.get("owner", "TODO-owner"),
        scope=ctx.track,
        evidence=[str(ctx.out("review_queue.csv")), str(ctx.out("ranked_review.csv"))],
        metrics={"n_items": len(final), "review_budget": queue_size, "max_score": round(top_score, 6)},
    )


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    # Keep the submission artifact exactly to the track contract; diagnostics stay in the review queue.
    final[spec.pred_columns].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
