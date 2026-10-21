"""Unit tests for test_evaluate module."""

from taed2_safeagent.modeling.test_evaluate import build_comparison


def test_build_comparison_structure(tmp_path):
    mock_svm = {
        "rows": 100,
        "macro_f1": 0.85,
        "accuracy": 0.86,
        "deny_to_allow_count": 5,
        "classes": {
            "ALLOW": {"precision": 0.88, "recall": 0.90},
            "ASK": {"precision": 0.82, "recall": 0.80},
            "DENY": {"precision": 0.85, "recall": 0.92},
        },
    }
    mock_bert = {
        "rows": 100,
        "macro_f1": 0.90,
        "accuracy": 0.91,
        "deny_to_allow_count": 2,
        "classes": {
            "ALLOW": {"precision": 0.93, "recall": 0.94},
            "ASK": {"precision": 0.88, "recall": 0.87},
            "DENY": {"precision": 0.89, "recall": 0.93},
        },
    }

    comp = build_comparison(mock_svm, mock_bert, output_dir=tmp_path)

    assert "models" in comp
    assert "tier1_edge_virtech" in comp["models"]
    assert "tier2_cloud_theoretical" in comp["models"]
    assert comp["models"]["tier1_edge_virtech"]["test_metrics"]["macro_f1"] == 0.85
    assert comp["models"]["tier2_cloud_theoretical"]["test_metrics"]["macro_f1"] == 0.90

    report_file = tmp_path / "test_evaluation_report.md"
    summary_file = tmp_path / "test_comparison_summary.json"

    assert report_file.is_file()
    assert summary_file.is_file()
    assert "Final Test Set Generalization Report" in report_file.read_text(encoding="utf-8")
