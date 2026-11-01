from types import SimpleNamespace

import pytest

from vericlick.title_promise import (
    analyze_title_promise,
    analyze_title_claims,
    extract_title_claims,
    extract_title_promise,
    extract_youtube_video_id,
    parse_timestamped_transcript,
    youtube_urls,
)


def test_extract_youtube_video_id_supported_url_forms():
    video_id = "abcdefghijk"
    assert extract_youtube_video_id(video_id) == video_id
    assert extract_youtube_video_id(f"https://www.youtube.com/watch?v={video_id}&t=8") == video_id
    assert extract_youtube_video_id(f"https://youtu.be/{video_id}") == video_id
    assert extract_youtube_video_id(f"https://www.youtube.com/shorts/{video_id}") == video_id
    assert youtube_urls(video_id)["embed_url"].endswith(f"/embed/{video_id}")


def test_reject_non_youtube_url():
    with pytest.raises(ValueError, match="Only YouTube"):
        extract_youtube_video_id("https://evil.example/watch?v=abcdefghijk")


def test_parse_timestamped_transcript_preserves_ranges():
    segments = parse_timestamped_transcript(
        "[00:12 - 00:20] Deep learning is a subset of machine learning.\n"
        "00:21: This segment explains neural network layers."
    )
    assert segments[0]["start"] == 12
    assert segments[0]["end"] == 20
    assert "Deep learning" in segments[0]["text"]
    assert segments[1]["start"] == 21


def test_untimed_transcript_does_not_invent_timestamps():
    segments = parse_timestamped_transcript("Deep learning uses neural networks. It learns representations from examples.")
    assert segments
    assert all(segment["start"] is None for segment in segments)


def test_vague_teaser_gets_not_enough_evidence():
    promise = extract_title_promise("YOU WON'T BELIEVE WHAT HAPPENED NEXT!!!")
    assert promise["kind"] == "vague_teaser"
    result = analyze_title_promise("YOU WON'T BELIEVE WHAT HAPPENED NEXT!!!", [
        {"start": 0, "end": 10, "text": "Welcome back. Subscribe and watch until the end."},
    ])
    assert result["fulfillment_status"] == "Not Enough Evidence"


def test_title_promise_identifies_definition_evidence_and_timestamp():
    segments = [
        {"start": 0.0, "end": 18.0, "text": "What is deep learning? Deep learning is a subset of machine learning inspired by the human brain."},
        {"start": 18.0, "end": 34.0, "text": "It uses layers of neural networks to learn patterns from examples. For example, image recognition can use deep neural networks."},
    ]
    result = analyze_title_promise("What is deep learning?", segments)
    assert result["extracted_promise"]
    assert result["fulfillment_status"] == "Fulfilled"
    assert result["evidence_timestamps"]
    assert result["evidence_timestamps"][0]["timestamp_range"].startswith("[")
    assert "learn" in result["matched_terms"]
    assert 0 <= result["topic_term_coverage_percent"] <= 100
    assert 0 <= result["alignment_score_percent"] <= 100
    assert result["llm_used"] is False
    assert "not a probability" in result["alignment_score_definition"]


def test_irrelevant_transcript_is_not_enough_evidence():
    result = analyze_title_promise("How to learn Python programming", [
        {"start": 10, "end": 24, "text": "The local weather is sunny, and birds are singing in the garden today."},
    ])
    assert result["fulfillment_status"] == "Not Enough Evidence"


def test_promise_result_has_required_disclaimer_fields():
    result = analyze_title_promise("Honest review of a laptop", [
        {"start": 0, "end": 20, "text": "In this review we tested battery life, display quality, keyboard and performance. The verdict is that the battery is strong, but the display is average."},
    ])
    assert result["fulfillment_status"] in {"Fulfilled", "Partly Fulfilled", "Not Enough Evidence"}
    assert "method" in result
    assert "reasoning" in result



def test_promiseproof_splits_numeric_free_and_outcome_promises():
    claims = extract_title_claims("5 free AI tools that save 10 hours")
    assert [claim["claim_type"] for claim in claims] == ["topic", "list_count", "free", "quantified_outcome"]
    assert claims[1]["required_count"] == 5
    assert claims[3]["amount"] == 10
    assert claims[3]["unit"] == "hours"


def test_promiseproof_distinguishes_support_from_mention_and_qualifiers():
    transcript = parse_timestamped_transcript(
        "[00:00 - 00:25] Today we look at five AI tools. We will show you five tools for productivity, including ChatGPT and other AI assistants.\n"
        "[00:25 - 00:50] Some of the tools have free plans, while other features require payment. You can test the free tier first.\n"
        "[00:50 - 01:20] These tools can save time, but savings depend on how you use them. This example will not guarantee saving ten hours every week.\n"
        "[01:20 - 01:50] First tool is ChatGPT, second is Claude, third is Gemini, fourth is Perplexity, and fifth is Copilot."
    )
    result = analyze_title_claims("5 free AI tools that save 10 hours", transcript)
    assert result["total_claims"] == 4
    by_type = {claim["claim_type"]: claim for claim in result["claims"]}
    assert by_type["topic"]["status"] == "Supported"
    assert by_type["list_count"]["status"] == "Supported"
    assert by_type["free"]["status"] == "Partly supported"
    assert by_type["quantified_outcome"]["status"] == "No evidence found"
    assert result["evidence_coverage_percent"] == 75
    assert result["fully_supported_percent"] == 50
    assert result["llm_used"] is False
    assert "not factual accuracy" in result["score_definition"]


def test_promiseproof_duration_keeps_the_core_topic_claim():
    claims = extract_title_claims("How to learn Python in 10 minutes")
    assert [claim["claim_type"] for claim in claims] == ["topic", "duration"]
    assert "python" in claims[0]["terms"]


def test_promiseproof_vague_teaser_is_not_made_into_fake_specific_claim():
    claims = extract_title_claims("You won't believe what happened next!!!")
    assert len(claims) == 1
    assert claims[0]["claim_type"] == "vague_teaser"
    result = analyze_title_claims("You won't believe what happened next!!!", [
        {"start": 0, "end": 20, "text": "Welcome back to the channel. Like and subscribe."},
    ])
    assert result["claims"][0]["status"] == "No evidence found"
    assert "too vague" in result["claims"][0]["reasoning"]
