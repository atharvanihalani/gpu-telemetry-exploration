"""Formatted display for classification results in notebooks."""

from IPython.display import display, HTML
from .rules import WindowVerdict


RULE_DESCRIPTIONS = {
    "power": "Mean power > {t.power:.0f}W (sustained high compute)",
    "tensor_ratio": "Tensor/SM ratio > {t.tensor_ratio:.2f} (matmul-dominated workload)",
    "power_std": "Power temporal std > {t.power_std:.0f}W (step-cycle oscillation)",
}


def _banner_html(detected: bool, n_flagged: int, n_total: int) -> str:
    if detected:
        color, bg = "#b91c1c", "#fef2f2"
        icon = "&#9632;"  # filled square
        label = "TRAINING DETECTED"
        detail = f"{n_flagged} of {n_total} windows flagged"
    else:
        color, bg = "#15803d", "#f0fdf4"
        icon = "&#9632;"
        label = "CLEAN"
        detail = f"0 of {n_total} windows flagged"

    return f"""
    <div style="border:2px solid {color}; border-radius:8px; padding:16px 20px;
                background:{bg}; font-family:monospace; margin:8px 0;">
        <span style="color:{color}; font-size:20px; font-weight:bold;">
            {icon} {label}
        </span>
        <span style="color:#555; font-size:14px; margin-left:16px;">{detail}</span>
    </div>
    """


def _window_row_html(v: WindowVerdict) -> str:
    if v.is_training:
        badge_color, badge_bg = "#b91c1c", "#fee2e2"
        badge = "FLAG"
    else:
        badge_color, badge_bg = "#15803d", "#dcfce7"
        badge = "CLEAN"

    rules_str = ", ".join(v.triggered_rules) if v.triggered_rules else "-"

    return f"""
    <tr style="border-bottom:1px solid #e5e7eb;">
        <td style="padding:6px 12px; font-family:monospace;">{v.start_s:.0f}–{v.end_s:.0f}s</td>
        <td style="padding:6px 12px; text-align:center;">
            <span style="background:{badge_bg}; color:{badge_color}; padding:2px 8px;
                         border-radius:4px; font-weight:bold; font-size:12px;">{badge}</span>
        </td>
        <td style="padding:6px 12px; font-family:monospace; text-align:right;">{v.mean_power:.0f}W</td>
        <td style="padding:6px 12px; font-family:monospace; text-align:right;">{v.tensor_sm_ratio:.3f}</td>
        <td style="padding:6px 12px; font-family:monospace; text-align:right;">{v.power_std_temporal:.1f}W</td>
        <td style="padding:6px 12px; font-family:monospace; color:#666;">{rules_str}</td>
    </tr>
    """


def _rules_reference_html(thresholds) -> str:
    lines = []
    for key, template in RULE_DESCRIPTIONS.items():
        lines.append(f"<li style='margin:4px 0; font-size:13px;'>"
                     f"<code>{key}</code>: {template.format(t=thresholds)}</li>")
    return f"""
    <details style="margin:12px 0; font-family:monospace;">
        <summary style="cursor:pointer; color:#555; font-size:13px;">
            Rule definitions (click to expand)
        </summary>
        <ul style="margin:8px 0;">{''.join(lines)}</ul>
        <p style="font-size:12px; color:#888; margin:4px 0 0 0;">
            A window is flagged if <b>any</b> rule triggers.
        </p>
    </details>
    """


def show_results(verdicts: list[WindowVerdict], thresholds=None):
    """Display classification results as formatted HTML in a notebook."""
    from .rules import Thresholds
    if thresholds is None:
        thresholds = Thresholds()

    n_flagged = sum(1 for v in verdicts if v.is_training)
    detected = n_flagged > 0

    # Banner
    html = _banner_html(detected, n_flagged, len(verdicts))

    # Rules reference
    html += _rules_reference_html(thresholds)

    # Per-window table
    html += """
    <table style="border-collapse:collapse; width:100%; font-size:14px; margin:8px 0;">
        <thead>
            <tr style="border-bottom:2px solid #999; text-align:left;">
                <th style="padding:6px 12px;">Window</th>
                <th style="padding:6px 12px; text-align:center;">Verdict</th>
                <th style="padding:6px 12px; text-align:right;">Power</th>
                <th style="padding:6px 12px; text-align:right;">Tensor/SM</th>
                <th style="padding:6px 12px; text-align:right;">Power Std</th>
                <th style="padding:6px 12px;">Triggered</th>
            </tr>
        </thead>
        <tbody>
    """
    for v in verdicts:
        html += _window_row_html(v)
    html += "</tbody></table>"

    display(HTML(html))
