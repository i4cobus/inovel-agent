| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session |
|---|---|---|---|---|---|---|---|---|---|
| overall | 75 | 0.2267 | 0.6636 | 6.9 | 50720 | 0.0214 | 0.3986 | 0.8148 | 0.9 |
| kind=constrained_rec | 50 | 0.28 | 0.74 | 8.36 | 62171 | 0.0125 | 0.4 | 0.8148 | 0.78 |
| kind=memory | 25 | 0.12 | 0.5965 | 5.61 | 40675 | 0.0282 | 0.3974 | None | 1.0 |
| variant=accumulate | 7 | 0.0 | 0.381 | 4.14 | 24134 | 0.0 | 0.1667 | None | 0.0 |
| variant=in_text | 16 | 0.25 | 0.75 | 7.69 | 64855 | 0.0208 | 0.3333 | None | 1.0 |
| variant=meta | 14 | 0.4286 | 0.8571 | 8.79 | 64851 | None | None | 0.8667 | 1.0 |
| variant=mixed | 20 | 0.2 | 0.65 | 8.6 | 58148 | 0.0 | 0.5 | 0.75 | 0.45 |
| variant=oneoff | 8 | 0.375 | 0.8125 | 7.0 | 60160 | 0.0 | 0.619 | None | 2.94 |
| variant=persist | 10 | 0.0 | 0.65 | 6.05 | 42454 | 0.0564 | 0.3846 | None | 0.5 |

failure labels: no_recommendation=32, tool_skipped=28, format=28, budget_exhausted=7, loop=5, constraint_violated=4, memory_missed=2, ungrounded_id=1
