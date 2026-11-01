from vericlick.media_analyzer import (
    estimate_content_specificity,
    estimate_context_match,
    extractive_summary,
    _tokens,
)


def test_extractive_summary_selects_up_to_requested_sentences():
    text = (
        "Python is a programming language used for automation and data analysis. "
        "This tutorial explains variables, loops, and functions with code examples. "
        "The weather outside is sunny and the birds are singing in the garden. "
        "You will practice a small Python program and learn how to debug errors. "
        "Finally, we review the program and recap the main concepts."
    )
    summary = extractive_summary(text, max_sentences=3)
    assert 1 <= len(summary) <= 3
    assert any("Python" in sentence for sentence in summary)


def test_context_match_uses_user_goal_terms_and_supports_devanagari_tokens():
    result = estimate_context_match(
        "This lesson teaches Python basics, functions, variables, and data analysis.",
        "I want to learn Python basics",
    )
    assert result["score"] >= 50
    assert "python" in result["matched_terms"]

    hindi_tokens = _tokens("मुझे Python सीखना है और उदाहरण चाहिए")
    assert "मुझे" in hindi_tokens
    assert "python" in hindi_tokens



def test_context_match_normalizes_learning_basics_and_ignores_exam_intent():
    result = estimate_context_match(
        "What exactly is deep learning? It's a subset of AI inspired by the human brain. "
        "By using layers of neural networks, it learns from self-driving cars, it's shaping our future.",
        "I want to learn deep learning basics for my exam.",
    )
    assert result["score"] >= 60
    assert "deep" in result["matched_terms"]
    assert "learn" in result["matched_terms"]
    assert "exam" not in result["goal_terms"]

def test_context_match_explains_missing_goal():
    result = estimate_context_match("Some long transcript words here", "")
    assert result["score"] is None
    assert "No viewing goal" in result["explanation"]


def test_specificity_score_is_bounded_and_disclosed_as_heuristic():
    result = estimate_content_specificity("First we compare two methods. For example, test the result and fix the code.")
    assert 0 <= result["score"] <= 100
    assert "heuristic" in result["basis"].lower()


def test_media_analyzer_builds_goal_based_report_with_stub_transcriber(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vericlick import media_analyzer

    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"fake media for stubbed integration")
    monkeypatch.setattr(media_analyzer, "_media_metadata_and_frames", lambda path: (125.0, True, True, []))

    class FakeModel:
        def transcribe(self, path, **kwargs):
            segments = [
                SimpleNamespace(start=0.0, end=20.0, text="This Python tutorial teaches variables and basic functions with practical code examples."),
                SimpleNamespace(start=20.0, end=55.0, text="First, we create a small program, then test the result and fix common errors."),
                SimpleNamespace(start=60.0, end=100.0, text="Finally, we review Python concepts and practice debugging the program."),
            ]
            info = SimpleNamespace(language="en", language_probability=0.99)
            return iter(segments), info

    monkeypatch.setattr(media_analyzer, "_load_whisper_model", lambda model_size: FakeModel())
    result = media_analyzer.analyze_media_file(
        media_path, "sample.mp4", title="Python tutorial", goal="I want to learn Python basics and functions",
    )
    assert result["has_video"] is True
    assert result["language"] == "en"
    assert result["context_match_score"] >= 50
    assert result["worth_score"] is not None
    assert result["summary"]
    assert result["key_moments"]
    assert result["title_promise_analysis"] is not None
    assert result["title_promise_analysis"]["fulfillment_status"] in {"Fulfilled", "Partly Fulfilled", "Not Enough Evidence"}
    assert "heuristic" in result["content_specificity_basis"].lower()
