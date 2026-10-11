| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage |
|---|---|---|---|---|---|---|---|---|
| overall | 20 | 0.7 | 0.931 | 6.34 | 45677 | 0.0 | 1.0 | 0.4286 |
| kind=constrained_rec | 14 | 0.5714 | 0.8571 | 8.14 | 64058 | 0.0 | 1.0 | 0.4286 |
| kind=memory | 6 | 1.0 | 1.0 | 4.67 | 28522 | 0.0 | 1.0 | None |
| variant=accumulate | 3 | 1.0 | 1.0 | 4.0 | 20692 | 0.0 | 1.0 | None |
| variant=in_text | 5 | 1.0 | 1.0 | 8.0 | 66694 | 0.0 | 1.0 | None |
| variant=meta | 5 | 0.4 | 1.0 | 8.2 | 65803 | None | None | 0.4 |
| variant=mixed | 4 | 0.25 | 0.5 | 8.25 | 58580 | 0.0 | 1.0 | 0.5 |
| variant=oneoff | 2 | 1.0 | 1.0 | 5.75 | 44415 | 0.0 | 1.0 | None |
| variant=persist | 1 | 1.0 | 1.0 | 5.5 | 31974 | 0.0 | 1.0 | None |

failure labels: tool_skipped=4, format=2, no_recommendation=2, budget_exhausted=1, model_error=1
