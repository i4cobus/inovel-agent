| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage |
|---|---|---|---|---|---|---|---|---|
| overall | 20 | 0.75 | 1.0 | 6.48 | 47133 | 0.0 | 0.875 | 0.8889 |
| kind=constrained_rec | 14 | 0.7143 | 1.0 | 7.57 | 60908 | 0.0 | 0.8519 | 0.8889 |
| kind=memory | 6 | 0.8333 | 1.0 | 5.47 | 34275 | 0.0 | 0.9048 | None |
| variant=accumulate | 3 | 0.6667 | 1.0 | 4.22 | 24158 | 0.0 | 0.7778 | None |
| variant=in_text | 5 | 0.8 | 1.0 | 7.4 | 63289 | 0.0 | 0.8 | None |
| variant=meta | 5 | 0.8 | 1.0 | 7.2 | 52856 | None | None | 0.8 |
| variant=mixed | 4 | 0.5 | 1.0 | 8.25 | 67999 | 0.0 | 0.9167 | 1.0 |
| variant=oneoff | 2 | 1.0 | 1.0 | 7.0 | 45628 | 0.0 | 1.0 | None |
| variant=persist | 1 | 1.0 | 1.0 | 8.0 | 57100 | 0.0 | 1.0 | None |

failure labels: tool_skipped=4, no_recommendation=1
