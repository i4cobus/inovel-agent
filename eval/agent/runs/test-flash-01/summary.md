| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session |
|---|---|---|---|---|---|---|---|---|---|
| overall | 75 | 0.8667 | 0.9813 | 6.06 | 42390 | 0.0119 | 0.9619 | 0.9495 | 2.14 |
| kind=constrained_rec | 50 | 0.86 | 0.98 | 6.96 | 51440 | 0.0238 | 0.981 | 0.9495 | 1.82 |
| kind=memory | 25 | 0.88 | 0.9825 | 5.26 | 34452 | 0.0 | 0.9429 | None | 2.42 |
| variant=accumulate | 7 | 0.8571 | 0.9524 | 3.95 | 20989 | 0.0 | 0.8929 | None | 0.57 |
| variant=in_text | 16 | 0.875 | 1.0 | 6.75 | 51146 | 0.0521 | 1.0 | None | 3.31 |
| variant=meta | 14 | 0.9286 | 1.0 | 6.71 | 51177 | None | None | 1.0 | 0.07 |
| variant=mixed | 20 | 0.8 | 0.95 | 7.3 | 51860 | 0.0 | 0.9649 | 0.9123 | 1.85 |
| variant=oneoff | 8 | 1.0 | 1.0 | 5.94 | 40304 | 0.0 | 1.0 | None | 3.5 |
| variant=persist | 10 | 0.8 | 1.0 | 6.1 | 43906 | 0.0 | 0.9375 | None | 3.5 |

failure labels: tool_skipped=6, constraint_violated=2, format=2, no_recommendation=2, model_error=1
