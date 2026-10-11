| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage |
|---|---|---|---|---|---|---|---|---|
| overall | 20 | 0.55 | 1.0 | 6.45 | 44794 | 0.0 | 0.875 | 0.713 |
| kind=constrained_rec | 14 | 0.5714 | 1.0 | 7.79 | 60151 | 0.0 | 0.9722 | 0.713 |
| kind=memory | 6 | 0.5 | 1.0 | 5.2 | 30461 | 0.0 | 0.75 | None |
| variant=accumulate | 3 | 0.3333 | 1.0 | 4.78 | 23966 | 0.0 | 0.5 | None |
| variant=in_text | 5 | 0.6 | 1.0 | 8.2 | 66933 | 0.0 | 0.95 | None |
| variant=meta | 5 | 0.6 | 1.0 | 7.2 | 55134 | None | None | 0.6 |
| variant=mixed | 4 | 0.5 | 1.0 | 8.0 | 57946 | 0.0 | 1.0 | 0.8542 |
| variant=oneoff | 2 | 0.5 | 1.0 | 5.25 | 36182 | 0.0 | 0.875 | None |
| variant=persist | 1 | 1.0 | 1.0 | 7.0 | 48248 | 0.0 | 1.0 | None |

failure labels: tool_skipped=8, no_recommendation=1
