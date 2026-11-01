"""Render audited RTC outcomes without confusing smoothness and task success.

Args: --report: completed formal RTC results; --output: owned figure directory.
Returns: PNG/SVG and exact plotted values with source SHA256.
Raises: Incomplete, mixed-family or missing-field errors.
Design: Context: Same-weight RTC comparisons have several competing outcomes.
    Concern: A smoother switch alone can conceal starvation or task failures.
    Decision: Show success, whole-worker latency, holds and matched old/new disagreement together.
    Alternatives: A single success or smoothness chart. Quality Attributes: Clarity, auditability.
    Trade-offs: Small descriptive samples; no broad model ranking.
Evidence: Internal: summarize_continuity formal1..5 results and separate overlap counts.
    External: RTC prefix-guided chunk completion. Findings: Queue-empty switches have no old next action.
    Metrics: Counts, milliseconds, percent and dimensionless OSC command difference.
    Confidence: Limited to the recorded task/configuration.
Constraints: Do not pool development or superseded GPU-sharing series.
Reconsider When: More tasks, repeats or a different control space are evaluated.
References: docs/experiments/realtime-continuity.md. ADR: None.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/pai-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(report,output):
    """Plot a complete one-family three-method grid and preserve the input values.

    Args: report: result JSON; output: figure folder. Returns: Plotted-data mapping.
    Raises: Invalid grid/status or missing metrics.
    Design: Context: Algorithm and resource cost must be assessed together.
        Concern: Overlap-only discontinuity has a different sample count from task outcomes.
        Decision: Label every denominator and retain queue-starvation alongside overlap values.
        Alternatives: Convert command differences to unmeasured physical trajectory errors.
        Quality Attributes: Honest comparison. Trade-offs: Four compact panels.
    Evidence: Internal: Complete5-state groups. External: None.
        Findings: Normed OSC commands have no direct physical-error unit.
        Metrics: Native success/5, worker ms, hold%, disagreement norm and n.
        Confidence: Descriptive only.
    Constraints: Same family, H50/R30, no selection of successful states.
    Reconsider When: Report schema or protocol changes.
    References: summarize_continuity.py. ADR: None.
    """
    report=Path(report);output=Path(output);data=json.loads(report.read_text())
    if data['status']!='complete':raise ValueError('complete report required')
    groups={g['variant']:g for g in data['groups']}
    order=['off','rtc-stock','rtc-vjp']
    if set(groups)!=set(order) or len({g['model'] for g in groups.values()})!=1:raise ValueError('one complete family required')
    if any(g['n']!=5 for g in groups.values()):raise ValueError('expected formal states1..5')
    rows=[]
    for key in order:
        g=groups[key]
        rows.append({'variant':key,'successes':g['successes'],'n':g['n'],
            'worker_ms':g['full_chunk_worker_roundtrip_ms_steady_episode_mean']['mean'],
            'hold_percent':g['pooled_hold_fraction']*100,
            'overlap_translation_disagreement':g['overlap_disagreement_translation_l2']['mean'],
            'overlap_n':g['overlap_disagreement_count']})
    family=next(iter(groups.values()))['model'];name={'smolvla':'SmolVLA','pi05':'pi0.5'}.get(family,family)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'#f8f7f1','axes.facecolor':'#f8f7f1','text.color':'#203630','axes.labelcolor':'#203630',
        'xtick.color':'#203630','ytick.color':'#203630'})
    fig,axes=plt.subplots(2,2,figsize=(11.2,6.6),layout='constrained')
    labels=['No RTC','LeRobot RTC','Full VJP RTC'];colors=['#9eaea1','#146b5c','#cc6c31'];x=np.arange(3)
    fields=['successes','worker_ms','hold_percent','overlap_translation_disagreement']
    titles=['Native task success','Generation + ROS','Empty-queue hold','Old vs new action at the same switch tick']
    ylabels=['Success count / 5','Mean per episode [ms]','Control ticks [%]','Translation-command difference [unitless]']
    for ax,field,title,ylabel in zip(axes.ravel(),fields,titles,ylabels):
        values=[r[field] for r in rows];observed=[v for v in values if v is not None];upper=max(observed,default=1)*1.35
        if field=='successes':upper=6
        elif field=='hold_percent':upper=max(30,upper)
        elif field=='overlap_translation_disagreement':upper=max(1.15,upper)
        for i,value in enumerate(values):
            if value is None:
                ax.text(i,upper*.04,f'n/a\n(n={rows[i]["overlap_n"]})',ha='center',va='bottom',fontsize=11)
                continue
            ax.bar(i,value,color=colors[i],width=.57)
            label=f'{int(value)}/5' if field=='successes' else f'{value:.0f}' if field=='worker_ms' else f'{value:.2f}%' if field=='hold_percent' else f'{value:.3f}\n(n={rows[i]["overlap_n"]})'
            ax.text(i,value+upper*.025,label,ha='center',va='bottom',fontsize=11,fontweight='bold')
        ax.set(xticks=x,xticklabels=labels,ylim=(0,upper),title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(f'{name} / same frozen weights / H50-R30 / 20 Hz / states 1–5',fontsize=14,fontweight='bold')
    output.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:fig.savefig(output/f'rtc-{family}.{extension}',dpi=160)
    plt.close(fig)
    values={'model':family,'rows':rows,'source_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
        'scope':'Formal5 per method; overlap disagreement excludes switches after empty queues and shows its own count. Command norm is not a measured EEF trajectory error.'}
    (output/f'rtc-{family}-plot-data.json').write_text(json.dumps(values,indent=2)+'\n');return values


def main():
    """Build figures only from explicitly selected audited data.

    Args: CLI report/output. Returns: None. Raises: Validation/render errors.
    Design: Context: Several historical series exist. Concern: Accidental pooling.
        Decision: One explicit source per invocation. Alternatives: Glob all runs.
        Quality Attributes: Reproducibility. Trade-offs: Caller selects final series.
    Evidence: Internal: Source hash in plotted data. External: Matplotlib.
        Findings: No model or simulator is loaded. Metrics: None. Confidence: IO contract.
    Constraints: No source-data writes. Reconsider When: Report schema changes.
    References: render. ADR: None.
    """
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();v=render(a.report,a.output);print(json.dumps({'model':v['model'],'figures':str(a.output),'rows':len(v['rows'])}))


if __name__=='__main__':main()
