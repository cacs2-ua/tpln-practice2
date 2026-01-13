import os
from pathlib import Path

import pytest
import torch

# Make matplotlib safe in headless pytest runs
import matplotlib
matplotlib.use("Agg")

from section10_visualization import (
    plot_logit_diff_heatmap,
    save_figure_publication_quality,
)


def test_section10_plot_returns_figure_and_axes_and_image():
    # shape matches assignment: 12 layers x 11 tokens
    m = torch.randn(12, 11)
    token_labels = [f" tok{i}" for i in range(11)]

    fig, ax = plot_logit_diff_heatmap(
        m,
        token_labels=token_labels,
        metric_title="Logit difference heatmap: logit(' B') − logit(' A')",
        show_token_strings=True,
        max_xticks=40,
        center_zero=True,
    )

    # should have at least one image in axis
    assert len(ax.images) == 1, "Expected a single heatmap image (matshow/imshow)."

    # should have colorbar axis as well (figure axes count >= 2)
    assert len(fig.axes) >= 2, "Expected a colorbar to be added."

    # title should be set
    assert "Logit difference" in ax.get_title()

    # x/y labels should be set
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""


def test_section10_save_publication_quality_creates_png_and_pdf(tmp_path: Path):
    m = torch.randn(12, 15)
    token_labels = [f" tok{i}" for i in range(15)]

    fig, ax = plot_logit_diff_heatmap(
        m,
        token_labels=token_labels,
        metric_title="Metric title",
        show_token_strings=True,
        max_xticks=40,
        center_zero=True,
    )

    out_base = tmp_path / "heatmap_test"
    saved = save_figure_publication_quality(fig, out_basepath=out_base, formats=("png", "pdf"), dpi=300)

    assert len(saved) == 2
    for p in saved:
        assert p.exists(), f"Expected saved file to exist: {p}"
        assert p.stat().st_size > 0, f"Expected saved file to be non-empty: {p}"


def test_section10_token_label_fallback_to_numeric_when_disabled():
    m = torch.randn(12, 60)
    token_labels = [f" tok{i}" for i in range(60)]

    fig, ax = plot_logit_diff_heatmap(
        m,
        token_labels=token_labels,
        metric_title="Metric title",
        show_token_strings=False,   # forces numeric labels
        max_xticks=12,              # reduce ticks
        center_zero=True,
    )

    # xtick labels should be numeric strings
    texts = [t.get_text() for t in ax.get_xticklabels()]
    # some may be empty depending on backend/layout; keep it robust:
    nonempty = [x for x in texts if x.strip() != ""]
    assert len(nonempty) > 0
    assert all(s.replace("-", "").isdigit() for s in nonempty), "Expected numeric x-tick labels in fallback mode."
