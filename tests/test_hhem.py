from sage.adapters.hhem import HHEMPredictionError, HallucinationDetector


def _detector_with_failing_predict() -> HallucinationDetector:
    detector = HallucinationDetector.__new__(HallucinationDetector)
    detector.threshold = 0.5
    detector._format_premise = lambda *args, **kwargs: "premise"

    def _raise_prediction_error(_pairs):
        raise HHEMPredictionError("HHEM prediction failed")

    detector._predict = _raise_prediction_error
    return detector


def test_check_explanation_marks_degraded_result_on_prediction_failure() -> None:
    detector = _detector_with_failing_predict()

    result = detector.check_explanation(["evidence"], "explanation")

    assert result.score == 0.0
    assert result.is_hallucinated is True
    assert result.degraded is True
    assert result.error_message == "HHEM prediction failed"


def test_check_claims_marks_each_claim_degraded_on_prediction_failure() -> None:
    detector = _detector_with_failing_predict()

    results = detector.check_claims(["evidence"], ["claim one", "claim two"])

    assert [result.claim for result in results] == ["claim one", "claim two"]
    assert all(result.score == 0.0 for result in results)
    assert all(result.is_hallucinated is True for result in results)
    assert all(result.degraded is True for result in results)
    assert all(result.error_message == "HHEM prediction failed" for result in results)


def test_check_batch_marks_each_result_degraded_on_prediction_failure() -> None:
    detector = _detector_with_failing_predict()

    results = detector.check_batch(
        [
            (["evidence one"], "first explanation"),
            (["evidence two"], "second explanation"),
        ]
    )

    assert [result.explanation for result in results] == [
        "first explanation",
        "second explanation",
    ]
    assert all(result.degraded is True for result in results)
    assert all(result.error_message == "HHEM prediction failed" for result in results)
