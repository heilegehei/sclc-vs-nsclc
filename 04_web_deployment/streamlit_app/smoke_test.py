from __future__ import annotations


from model_runtime import FROZEN_DECISION_THRESHOLD, input_defaults, load_model, score_one


def main() -> None:
    model = load_model()
    result, contributions, audit = score_one(input_defaults())
    assert 0.0 <= float(result["weighted_voting_probability"]) <= 1.0
    assert abs(float(contributions["Weight"].sum()) - 1.0) < 1e-12
    assert audit["remaining_missing_n"] == 0
    print(
        {
            "status": "PASS",
            "model_name": model["model_name"],
            "probability": round(float(result["weighted_voting_probability"]), 8),
            "threshold": FROZEN_DECISION_THRESHOLD,
            "members": len(contributions),
        }
    )


if __name__ == "__main__":
    main()
