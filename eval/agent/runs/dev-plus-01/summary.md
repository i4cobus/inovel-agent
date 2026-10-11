| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage |
|---|---|---|---|---|---|---|---|---|
| overall | 20 | 0.3 | 0.7931 | 6.97 | 49897 | 0.0 | 0.7949 | 0.125 |
| kind=constrained_rec | 14 | 0.2857 | 0.8571 | 8.43 | 63958 | 0.0 | 0.7619 | 0.125 |
| kind=memory | 6 | 0.3333 | 0.7333 | 5.6 | 36773 | 0.0 | 0.8333 | None |
| variant=accumulate | 3 | 0.3333 | 0.6667 | 4.44 | 23399 | 0.0 | 1.0 | None |
| variant=in_text | 5 | 0.6 | 0.8 | 8.6 | 63982 | 0.0 | 0.75 | None |
| variant=meta | 5 | 0.2 | 1.0 | 8.8 | 70848 | None | None | 0.2 |
| variant=mixed | 4 | 0.0 | 0.75 | 7.75 | 55317 | 0.0 | 0.7778 | 0.0 |
| variant=oneoff | 2 | 0.5 | 1.0 | 6.25 | 51070 | 0.0 | 0.5 | None |
| variant=persist | 1 | 0.0 | 0.5 | 9.5 | 68365 | 0.0 | 1.0 | None |

failure labels: tool_skipped=9, format=5, budget_exhausted=3, no_recommendation=3
