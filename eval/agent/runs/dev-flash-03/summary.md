| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session |
|---|---|---|---|---|---|---|---|---|---|
| overall | 20 | 0.55 | 1.0 | 5.34 | 37829 | 0.0 | 0.8526 | 0.8333 | 1.93 |
| kind=constrained_rec | 14 | 0.5 | 1.0 | 6.0 | 44856 | 0.0 | 0.8333 | 0.8333 | 2.21 |
| kind=memory | 6 | 0.6667 | 1.0 | 4.73 | 31270 | 0.0 | 0.869 | None | 1.67 |
| variant=accumulate | 3 | 0.6667 | 1.0 | 3.89 | 23566 | 0.0 | 0.7778 | None | 0.56 |
| variant=in_text | 5 | 0.6 | 1.0 | 5.6 | 45497 | 0.0 | 0.75 | None | 1.8 |
| variant=meta | 5 | 0.4 | 1.0 | 6.2 | 42826 | None | None | 0.75 | 1.4 |
| variant=mixed | 4 | 0.5 | 1.0 | 6.25 | 46594 | 0.0 | 1.0 | 1.0 | 3.75 |
| variant=oneoff | 2 | 0.5 | 1.0 | 4.75 | 32142 | 0.0 | 0.875 | None | 4.0 |
| variant=persist | 1 | 1.0 | 1.0 | 8.5 | 64192 | 0.0 | 1.0 | None | 2.0 |

failure labels: tool_skipped=5, no_recommendation=4, ungrounded_id=1
