| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session |
|---|---|---|---|---|---|---|---|---|---|
| overall | 20 | 0.8 | 1.0 | 6.07 | 43660 | 0.0 | 0.9844 | 0.963 | 2.07 |
| kind=constrained_rec | 14 | 0.7857 | 1.0 | 6.93 | 53869 | 0.0 | 1.0 | 0.963 | 3.0 |
| kind=memory | 6 | 0.8333 | 1.0 | 5.27 | 34132 | 0.0 | 0.9643 | None | 1.2 |
| variant=accumulate | 3 | 0.6667 | 1.0 | 4.44 | 26191 | 0.0 | 0.9167 | None | 0.78 |
| variant=in_text | 5 | 0.8 | 1.0 | 7.8 | 63387 | 0.0 | 1.0 | None | 4.2 |
| variant=meta | 5 | 0.8 | 1.0 | 5.8 | 41855 | None | None | 0.9333 | 1.2 |
| variant=mixed | 4 | 0.75 | 1.0 | 7.25 | 56990 | 0.0 | 1.0 | 1.0 | 3.75 |
| variant=oneoff | 2 | 1.0 | 1.0 | 6.5 | 46180 | 0.0 | 1.0 | None | 2.0 |
| variant=persist | 1 | 1.0 | 1.0 | 6.5 | 45772 | 0.0 | 1.0 | None | 1.5 |

failure labels: no_recommendation=2, tool_skipped=2
