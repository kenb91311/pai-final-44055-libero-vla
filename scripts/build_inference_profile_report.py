"""Build descriptive frozen-inference tables and figures from JSON reports only.

Args: Input directory containing three native report.json files and a fresh output path.
Returns: Summary/plot-data JSON, total/stage CSV, and PNG/SVG figures.
Raises: ValueError for malformed or inconsistent timing accounting; IO errors propagate.
Design: Context: Native VLA horizons and precision differ in this installation.
    Concern: Nested stage sums or pooled baselines can create false speed explanations.
    Decision: Recompute per-condition statistics, preserve inclusive/exclusive metrics,
        and plot each measurement separately with exact source hashes.
    Alternatives: Add inclusive stages, infer kernel time from CUDA events, or load models.
    Quality Attributes: Traceability, isolation. Trade-offs: Descriptive small-sample results.
Evidence: Internal: profile_inference.py schema_version1 and preregistered protocol.
    External: None. Findings: Actual measured values are supplied by source reports.
    Metrics: Mean/median/p95/min/max milliseconds; independent exclusive partition checks.
    Confidence: JSON accounting tests; no model-level causal claim.
Constraints: JSON inputs only; no GPU, checkpoint, NPZ or historical rollout modifications.
Reconsider When: Profiling schema or native interval hierarchy changes.
References: docs/experiments/realtime-inference-profiling.md. ADR: None.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("smolvla", "pi05", "pi0fast")
DISPLAY = {"smolvla": "SmolVLA", "pi05": "pi0.5", "pi0fast": "pi0-FAST"}
HORIZONS = {"smolvla": 50, "pi05": 50, "pi0fast": 10}
CONDITIONS = ("baseline_before", "instrumented", "baseline_after")
METRICS = ("host_ms", "host_exclusive_ms", "cuda_event_elapsed_ms", "cuda_event_exclusive_ms")
STATISTICS = ("count", "mean", "median", "p95", "min", "max")
STAGE_ORDER = (
    "request_inclusive", "observation_preparation_inclusive", "libero_camera_rotation_and_state",
    "saved_preprocessing", "policy_predict_inclusive", "image_resize_and_policy_preparation",
    "action_sampling_inclusive", "autoregressive_sampling_inclusive", "prefix_embedding_inclusive",
    "camera_embedding_batch_inclusive", "vision_embedding_inclusive", "vision_tower", "vision_projector",
    "vlm_prefix_prefill", "flow_denoise_step", "flow_action_expert_forward",
    "fast_cached_decode_forward", "fast_logits_projection", "action_detokenization_inclusive",
    "fast_action_token_decode", "saved_postprocessing",
)


def describe(values):
    """Describe finite measured samples using a linearly interpolated sample percentile.

    Args: values: numeric observations in one stated unit. Returns: Count/mean/median/p95/min/max.
    Raises: ValueError for missing or nonfinite observations.
    Design: Context: Each native block has25 measured requests. Concern: Implied population precision.
        Decision: Descriptive summaries only; p95 uses position0.95*(n-1), linear interpolation.
        Alternatives: Confidence claims from five repeated scenes.
        Quality Attributes: Reproducibility. Trade-offs: No uncertainty estimate.
    Evidence: Internal: Raw JSON calls. External: None. Findings: No warmup values are supplied.
        Metrics: Count and scalar summaries. Confidence: Exact arithmetic definition.
    Constraints: Never pool different conditions. Reconsider When: Statistical inference is requested.
    References: summarize_report. ADR: None.
    """
    rows = sorted(float(value) for value in values)
    if not rows or not all(math.isfinite(value) for value in rows):
        raise ValueError("summary requires nonempty finite values")
    position = .95 * (len(rows) - 1)
    low, high = math.floor(position), math.ceil(position)
    p95 = rows[low] + (rows[high] - rows[low]) * (position - low)
    return {"count": len(rows), "mean": statistics.mean(rows), "median": statistics.median(rows),
            "p95": p95, "min": rows[0], "max": rows[-1]}


def source_record(path):
    """Fingerprint a source file without importing its contents as code.

    Args: path: existing source artifact. Returns: Path, SHA256 and file size.
    Raises: IO errors. Design: Context: Figures need precise reproducibility.
        Concern: Paths alone do not identify contents. Decision: Hash exact source bytes.
        Alternatives: Modification dates. Quality Attributes: Traceability. Trade-offs: Disk reads.
    Evidence: Internal: plot-data provenance. External: SHA256. Findings: None.
        Metrics: Bytes and digest. Confidence: Content identity.
    Constraints: Read-only. Reconsider When: Inputs become remote artifacts.
    References: build_report. ADR: None.
    """
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"file": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def summarize_report(path, expected_model):
    """Recompute totals and per-request stage statistics while verifying exclusive accounting.

    Args: path: native JSON report; expected_model: expected runtime identity.
    Returns: Model summary with metadata, unchanged validation, eligibility and stage summaries.
    Raises: ValueError for unsupported schema, pairing errors or invalid nested partitions.
    Design: Context: Inclusive stage labels overlap hierarchically.
        Concern: Double counting or interpreting CPU host waits as GPU work.
        Decision: Sum only same-label instances per request; verify all exclusive labels
            partition the root separately for host and CUDA-stream elapsed; exclude failed
            output/frozen checks from primary plots without altering the source flags.
        Alternatives: Sum inclusive labels or add host and CUDA time together.
        Quality Attributes: Accounting integrity. Trade-offs: Existing recorder tolerance0.1ms.
    Evidence: Internal: IntervalRecorder.finish accounting fields. External: None.
        Findings: Stage values are per-request totals, not average individual invocation time.
        Metrics: Descriptive stage totals, invocation counts and partition errors.
        Confidence: Numeric consistency only; source audit flags remain visible.
    Constraints: Warmup/probe excluded; invalid stage partitions fail before artifact creation.
    Reconsider When: Concurrent CUDA streams or missing interval stages are supported.
    References: profile_inference.py. ADR: None.
    """
    source = json.loads(Path(path).read_text())
    if source.get("schema_version") != 1 or source.get("metadata", {}).get("model") != expected_model:
        raise ValueError(f"unsupported schema or wrong model in {path}")
    blocks = source["conditions"]
    totals, latencies = {}, {}
    reference_keys = None
    for condition in CONDITIONS:
        calls = blocks[condition]["calls"]
        keys = [(row["sample_id"], row["repeat"], row["seed"], row["sensor_hashes"]) for row in calls]
        if reference_keys is not None and keys != reference_keys:
            raise ValueError(f"condition input/seed pairing differs: {path}")
        reference_keys = keys
        latencies[condition] = [float(row["wall_ms"]) for row in calls]
        if any(value < 0 for value in latencies[condition]):
            raise ValueError("negative request wall time")
        totals[condition] = describe(latencies[condition])
    measured = blocks["instrumented"]["calls"]
    labels = sorted({label for row in measured for label in row["profile"]["stages"]},
                    key=lambda label: (STAGE_ORDER.index(label) if label in STAGE_ORDER else len(STAGE_ORDER), label))
    stage_summary = {}
    for label in labels:
        stages = [row["profile"]["stages"].get(label, {}) for row in measured]
        stage_summary[label] = {"native_invocations_per_request": describe([stage.get("calls", 0) for stage in stages]),
                                "per_request_total_ms": {}}
        for metric in METRICS:
            if not any(metric in stage for stage in stages):
                continue
            values = [stage.get(metric, 0.) for stage in stages]
            stage_summary[label]["per_request_total_ms"][metric] = describe(values)
    partition_errors = {"host_ms": [], "cuda_stream_ms": []}
    for row in measured:
        profile = row["profile"]
        account = profile["accounting"]
        if not account.get("valid"):
            raise ValueError(f"source has invalid nested accounting: {path}")
        stages = profile["stages"]
        for metric, root_key, output_key in (
                ("host_exclusive_ms", "host_root_ms", "host_ms"),
                ("cuda_event_exclusive_ms", "cuda_root_event_elapsed_ms", "cuda_stream_ms")):
            if root_key not in account:
                if any(metric in stage for stage in stages.values()):
                    raise ValueError(f"missing root accounting for {metric}")
                continue
            values = [float(stage[metric]) for stage in stages.values()]
            difference = sum(values) - float(account[root_key])
            if (not all(math.isfinite(value) and value >= -.1 for value in values)
                    or not math.isfinite(difference) or abs(difference) > .1):
                raise ValueError(f"exclusive partition does not equal root: {metric}, {difference}")
            partition_errors[output_key].append(difference)
    before, instrumented, after = (latencies[name] for name in CONDITIONS)
    overhead_ms = [value - (a + b) / 2 for a, value, b in zip(before, instrumented, after)]
    shapes = {}
    for row in measured:
        shape = "x".join(str(value) for value in row["action_shape"])
        shapes[shape] = shapes.get(shape, 0) + 1
    if set(shapes) != {f"{HORIZONS[expected_model]}x7"}:
        raise ValueError(f"native action shape differs from plotted H{HORIZONS[expected_model]}: {shapes}")
    required_checks = ("all_nested_interval_accounting_valid", "all_paired_outputs_allclose", "weights_unchanged")
    exclusion_reasons = [f"source validation {key} is not true" for key in required_checks
                         if source["validation"].get(key) is not True]
    return {"model": expected_model, "source": source_record(path), "protocol": source["protocol"],
            "metadata": source["metadata"], "validation": source["validation"],
            "paired_comparison_eligible": not exclusion_reasons, "exclusion_reasons": exclusion_reasons,
            "frozen_audit": source.get("frozen_audit"), "paired": source.get("paired"),
            "memory": source.get("memory"), "observed_action_shapes": shapes,
            "native_horizon_label": HORIZONS[expected_model], "total_wall_ms": totals,
            "baseline_pooled_reference_wall_ms": describe(before + after),
            "baseline_pooled_reference_semantics": "descriptive concatenation of uninstrumented before/after calls; block drift remains visible separately",
            "raw_total_wall_ms": latencies, "stages": stage_summary,
            "paired_instrumented_minus_bracket_ms": describe(overhead_ms),
            "partition_errors_ms": {key: describe(values) if values else None for key, values in partition_errors.items()}}


def write_csv(path, rows, columns):
    """Write numeric table rows with an explicit stable column order.

    Args: path: output CSV; rows: mappings; columns: ordered column names.
    Returns: None. Raises: IO/schema errors.
    Design: Context: Summary tables support independent inspection.
        Concern: Implicit units or unstable dict columns. Decision: Explicit metric/unit/stat fields.
        Alternatives: Unlabelled numeric matrices. Quality Attributes: Portability. Trade-offs: Long stage table.
    Evidence: Internal: Generated summary.json values. External: CSV. Findings: None.
        Metrics: Same full-precision floats as JSON. Confidence: Serialization only.
    Constraints: No inferred values. Reconsider When: Table schema changes.
    References: build_report. ADR: None.
    """
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_figures(models, output, excluded_models=()):
    """Render total and stage figures and return exactly the plotted numerical data.

    Args: models: eligible native summaries; output: report directory;
        excluded_models: failed native summaries retained in tables, excluded from plots.
    Returns: Figure descriptions/data with source report digests and bar coordinates.
    Raises: Matplotlib/IO errors.
    Design: Context: Native full-chunk output is H50/H50/H10.
        Concern: Apparent one-action normalization or stacked overlapping intervals.
        Decision: Separate total statistic panels and separate host/CUDA inclusive/exclusive
            stage panels; primary plots only contain models passing paired source checks.
        Alternatives: Stacked inclusive bars or divide total latency by output horizon.
        Quality Attributes: Interpretability, reproducibility. Trade-offs: Three detailed figures.
    Evidence: Internal: Recomputed raw-call summaries. External: Matplotlib.
        Findings: CUDA events include possible host launch/idle gaps, not kernel busy time.
        Metrics: Exact bar values in plot-data.json. Confidence: Descriptive visualization.
    Constraints: Host and CUDA series never added; missing CUDA metrics are labelled unavailable.
    Reconsider When: More models or many additional stage labels are introduced.
    References: summarize_report. ADR: None.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#537c99", "#d3813b", "#60966f")
    model_labels = [f"{DISPLAY[model['model']]}\nH{model['native_horizon_label']} full chunk" for model in models]
    plot_data = {"source_reports": [model["source"] for model in models], "figures": {}}
    excluded_note = (" / Excluded: " + ", ".join(DISPLAY[model["model"]] for model in excluded_models)
                     + " (paired output check failed)") if excluded_models else ""
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    total_rows = []
    for axis, statistic in zip(axes, ("mean", "median", "p95")):
        for condition_index, condition in enumerate(CONDITIONS):
            values = [model["total_wall_ms"][condition][statistic] for model in models]
            positions = [index + (condition_index - 1) * .23 for index in range(len(models))]
            bars = axis.bar(positions, values, width=.22, color=colors[condition_index],
                            label=condition.replace("_", " "))
            axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
            total_rows.append({"statistic": statistic, "condition": condition, "positions": positions,
                               "values_ms": values, "models": [model["model"] for model in models]})
        axis.set(xticks=range(len(models)), xticklabels=model_labels, ylabel="Request wall elapsed (ms)",
                 title=f"{statistic} / measured requests only")
        axis.margins(y=.18)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    figure.suptitle("Validated native full-chunk inference / separate before, instrumented, after blocks\n"
                   + excluded_note.lstrip(" /"), fontsize=13)
    for extension in ("png", "svg"):
        figure.savefig(output / f"native-fullchunk.{extension}", dpi=170)
    plt.close(figure)
    plot_data["figures"]["native-fullchunk"] = {"unit": "ms", "bars": total_rows,
                                                 "model_labels": model_labels, "no_horizon_normalization": True}
    for kind, metric_pair in (("host", ("host_ms", "host_exclusive_ms")),
                              ("cuda-stream", ("cuda_event_elapsed_ms", "cuda_event_exclusive_ms"))):
        figure, axes = plt.subplots(len(models), 2, figsize=(20, 7.3 * len(models)),
                                    constrained_layout=True, squeeze=False)
        rows = []
        for row_index, model in enumerate(models):
            stages = list(model["stages"])
            for column, metric in enumerate(metric_pair):
                axis = axes[row_index, column]
                labels = [stage for stage in stages if metric in model["stages"][stage]["per_request_total_ms"]]
                values = [model["stages"][stage]["per_request_total_ms"][metric]["mean"] for stage in labels]
                accounting = "inclusive (nested; do not sum)" if column == 0 else "exclusive (direct children subtracted)"
                if labels:
                    bars = axis.barh(range(len(labels)), values, color=colors[row_index], height=.68)
                    axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
                    axis.set_yticks(range(len(labels)), labels, fontsize=9)
                    axis.invert_yaxis()
                    axis.margins(x=.18)
                else:
                    axis.text(.5, .5, "Metric unavailable in source report", ha="center", transform=axis.transAxes)
                    axis.set_yticks([])
                axis.set(xlabel="Mean per-request stage total (ms)",
                         title=f"{DISPLAY[model['model']]} H{model['native_horizon_label']} / {accounting}")
                axis.grid(axis="x", alpha=.2)
                axis.set_axisbelow(True)
                rows.append({"model": model["model"], "metric": metric, "statistic": "mean",
                             "stages": labels, "values_ms": values})
        meaning = ("Host elapsed includes CPU work, launches and implicit waits" if kind == "host" else
                   "CUDA stream elapsed includes possible host launch/idle gaps; not kernel busy time")
        figure.suptitle(f"Instrumented native stages / {meaning}\nSame-label calls are totalled within each request; panels have independent scales{excluded_note}",
                         fontsize=14)
        for extension in ("png", "svg"):
            figure.savefig(output / f"stage-{kind}.{extension}", dpi=170)
        plt.close(figure)
        plot_data["figures"][f"stage-{kind}"] = {"unit": "ms", "panels": rows, "semantics": meaning}
    return plot_data


def build_report(input_dir, output, ablation_report=None, protocol_path=None):
    """Build a fresh report bundle from three explicit native source folders.

    Args: input_dir: parent of MODEL-native/report.json; output: fresh destination;
        ablation_report: optional separate pi05 JSON; protocol_path: preregistration JSON.
    Returns: Summary mapping. Raises: FileExistsError for output reuse; source/accounting errors.
    Design: Context: Completed task-success series retain independent historical meaning.
        Concern: Accidental discovery/pooling of unrelated runs or synthetic fixture publication.
        Decision: Explicit native directory names; optional ablation stays separate; hash every source.
            Preserve failed reports in marked tables while excluding them from primary plots.
        Alternatives: Recursive glob across artifacts. Quality Attributes: Scope integrity.
        Trade-offs: Caller selects one finalized series.
    Evidence: Internal: inference-profile-protocol.json. External: None.
        Findings: Missing required native reports fail before output creation.
        Metrics: Three condition summaries per model and raw stage descriptive statistics.
        Confidence: Limited to supplied JSON/audit evidence.
    Constraints: No model/NPZ/GPU access; no modification of existing outputs.
    Reconsider When: A new protocol or repeated independent series requires comparison.
    References: render_figures. ADR: None.
    """
    input_dir, output = Path(input_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol_path = Path(protocol_path or ROOT / "artifacts/realtime-vla/inference-profile-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    models = [summarize_report(input_dir / f"{model}-native" / "report.json", model) for model in MODELS]
    auto_ablation = input_dir / "pi05-per-camera" / "report.json"
    ablation_path = Path(ablation_report) if ablation_report is not None else auto_ablation
    ablation = summarize_report(ablation_path, "pi05") if ablation_path.is_file() else None
    if ablation_report is not None and ablation is None:
        raise FileNotFoundError(ablation_path)
    manifests = [model["protocol"].get("observation_manifest_sha256") for model in models]
    sensor_sources = [model["protocol"].get("sample_provenance") for model in models]
    shared_inputs = (bool(manifests[0]) and len(set(manifests)) == 1
                     and bool(sensor_sources[0]) and all(value == sensor_sources[0] for value in sensor_sources))
    eligible_models = [model for model in models if model["paired_comparison_eligible"] and shared_inputs]
    excluded_models = [model for model in models if model not in eligible_models]
    if not eligible_models:
        raise ValueError("no models with shared inputs and valid paired source checks for primary plots")
    summary = {"schema_version": 1, "status": "partial_validated" if excluded_models else "source_checks_passed",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Offline frozen native inference; five initial observations repeated five times, not closed-loop task evaluation",
               "protocol": protocol, "sources": {"protocol": source_record(protocol_path),
                                                   "builder": source_record(__file__)},
               "statistics": {"unit": "milliseconds except invocation counts and explicitly named ratios",
                              "p95": "linear interpolation at sorted position 0.95*(n-1)",
                              "warmup_and_precision_probe": "excluded",
                              "stage_aggregation": "same-label invocation sums within each request, then describe requests",
                              "exclusive": "immediate-child subtraction; separately partitions host or CUDA-stream root",
                              "inclusive": "nested overlapping intervals; never add across stage labels",
                              "cuda": "single-stream event elapsed including possible CPU launch/idle gaps; not kernel busy time"},
               "shared_native_input_manifest_and_sample_provenance": shared_inputs,
               "primary_comparison": {"included_models": [model["model"] for model in eligible_models],
                                      "excluded_models": [{"model": model["model"], "reasons": model["exclusion_reasons"],
                                                            "source": model["source"]} for model in excluded_models],
                                      "policy": "failed source validity remains unchanged; excluded rows are descriptive reference only"},
               "models": models, "optional_pi05_per_camera_ablation": ablation,
               "limitations": ["Native H50/H50/H10 are full-chunk outputs, not equal action-generation work.",
                               "n_action_steps=1 does not turn SmolVLA native chunk generation into H1.",
                               "Stored parameter dtype and representative activation/autocast probes describe different quantities.",
                               "Model size, source implementations, image/token shapes, precision and samplers vary together.",
                               "Stage timing is instrumented; use both uninstrumented baselines to assess overhead and drift.",
                               "Repeated initial observations do not establish closed-loop success or a permanent model speed ranking."]}
    output.mkdir(parents=True, exist_ok=False)
    total_rows, stage_rows = [], []
    for group, rows in (("native", models), ("pi05_per_camera_ablation", [ablation] if ablation else [])):
        for model in rows:
            eligibility = {"paired_comparison_eligible": model["paired_comparison_eligible"],
                           "exclusion_reasons": "; ".join(model["exclusion_reasons"])}
            labelled_group = group if model["paired_comparison_eligible"] else f"{group}_excluded_reference"
            for condition, values in {**model["total_wall_ms"], "baseline_pooled_reference": model["baseline_pooled_reference_wall_ms"]}.items():
                total_rows.append({"group": labelled_group, "model": model["model"], "condition": condition,
                                   **eligibility,
                                   "native_horizon_label": model["native_horizon_label"], "unit": "ms", **values})
            for stage, values in model["stages"].items():
                for metric, statistics_row in values["per_request_total_ms"].items():
                    stage_rows.append({"group": labelled_group, "model": model["model"], "stage": stage, **eligibility,
                                       "metric": metric, "unit": "ms", **statistics_row})
                stage_rows.append({"group": labelled_group, "model": model["model"], "stage": stage, **eligibility,
                                   "metric": "native_invocations_per_request", "unit": "calls",
                                   **values["native_invocations_per_request"]})
    write_csv(output / "totals.csv", total_rows,
              ("group", "model", "paired_comparison_eligible", "exclusion_reasons", "native_horizon_label", "condition", "unit", *STATISTICS))
    write_csv(output / "stages.csv", stage_rows,
              ("group", "model", "paired_comparison_eligible", "exclusion_reasons", "stage", "metric", "unit", *STATISTICS))
    plot_data = render_figures(eligible_models, output, excluded_models)
    plot_data["sources"] = summary["sources"]
    plot_data["status"] = summary["status"]
    plot_data["primary_comparison"] = summary["primary_comparison"]
    (output / "plot-data.json").write_text(json.dumps(plot_data, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    """Expose explicit report sources and a fresh output destination through the CLI.

    Args: argv: optional argument sequence. Returns: None. Raises: Validation/render errors.
    Design: Context: Root finalizes this after serial GPU profiling.
        Concern: Importing the report command must never initialize model dependencies.
        Decision: Standard-library JSON aggregation and lazy CPU plotting only.
        Alternatives: Reuse model runtime loaders. Quality Attributes: Isolation. Trade-offs: None.
    Evidence: Internal: CPU help/temporary synthetic bundle smoke. External: None.
        Findings: No measured values are invented. Metrics: Output paths and model count.
        Confidence: Command contract.
    Constraints: Existing output refused. Reconsider When: Report resume is explicitly required.
    References: build_report. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/realtime-vla/inference-profile-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/realtime-vla/inference-profile-report")
    parser.add_argument("--ablation-report", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)
    summary = build_report(args.input_dir, args.output, args.ablation_report, args.protocol)
    print(json.dumps({"status": summary["status"], "models": len(summary["models"]),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
