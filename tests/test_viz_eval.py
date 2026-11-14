"""
Unit tests for solarflare.viz_eval module.
Verifies figures F12 to F21 render cleanly, adhere to styles, and generate takeaway boxes.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

from solarflare.viz_eval import (
    get_takeaway_box,
    add_takeaway_box,
    plot_f12_model_comparison,
    plot_f13_roc_pr_curves,
    plot_f14_confusion_matrix,
    plot_f15_threshold_sweep,
    plot_f16_reliability_diagram,
    plot_f17_skill_vs_horizon,
    plot_f18_ablation_delta_hss,
    plot_f19_permutation_importance,
    plot_f20_test_case_study,
    plot_f21_error_analysis,
    generate_all_eval_figures,
    TAKEAWAY_BOXES,
)


def test_takeaway_boxes_complete():
    """Verify all 10 figures have 3-line Markdown takeaway boxes."""
    for i in range(12, 22):
        f_id = f"F{i}"
        box = get_takeaway_box(f_id)
        assert "> **What you see:**" in box
        assert "> **Why it matters:**" in box
        assert "> **Say this in the viva:**" in box


def test_plot_f12_model_comparison(tmp_path):
    """Test F12 model comparison grouped bar rendering."""
    out_file = str(tmp_path / "F12.png")
    fig = plot_f12_model_comparison(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f13_roc_pr_curves(tmp_path):
    """Test F13 ROC and PR curves rendering."""
    out_file = str(tmp_path / "F13.png")
    fig = plot_f13_roc_pr_curves(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f14_confusion_matrix(tmp_path):
    """Test F14 confusion matrix rendering."""
    out_file = str(tmp_path / "F14.png")
    fig = plot_f14_confusion_matrix(tp=100, fn=20, fp=15, tn=800, save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f15_threshold_sweep(tmp_path):
    """Test F15 validation threshold sweep curve rendering."""
    out_file = str(tmp_path / "F15.png")
    fig = plot_f15_threshold_sweep(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f16_reliability_diagram(tmp_path):
    """Test F16 reliability diagram and prediction histogram rendering."""
    out_file = str(tmp_path / "F16.png")
    fig = plot_f16_reliability_diagram(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f17_skill_vs_horizon(tmp_path):
    """Test F17 skill vs horizon curves rendering."""
    out_file = str(tmp_path / "F17.png")
    fig = plot_f17_skill_vs_horizon(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f18_ablation_delta_hss(tmp_path):
    """Test F18 clue ablation delta HSS rendering."""
    out_file = str(tmp_path / "F18.png")
    fig = plot_f18_ablation_delta_hss(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f19_permutation_importance(tmp_path):
    """Test F19 permutation importance horizontal bar chart rendering."""
    out_file = str(tmp_path / "F19.png")
    fig = plot_f19_permutation_importance(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f20_test_case_study(tmp_path):
    """Test F20 test period case study ('money plot') rendering."""
    out_file = str(tmp_path / "F20.png")
    fig = plot_f20_test_case_study(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_plot_f21_error_analysis(tmp_path):
    """Test F21 error analysis plot rendering."""
    out_file = str(tmp_path / "F21.png")
    fig = plot_f21_error_analysis(save=True, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_generate_all_eval_figures(tmp_path):
    """Test batch generation of all 10 evaluation figures."""
    save_dir = str(tmp_path / "figures")
    generated = generate_all_eval_figures(save_dir=save_dir)
    assert len(generated) == 10
    for fid in [f"F{i}" for i in range(12, 22)]:
        assert fid in generated
        assert os.path.exists(generated[fid])
