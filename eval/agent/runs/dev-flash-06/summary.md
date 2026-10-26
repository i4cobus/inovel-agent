| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session | unrequested_offtopic_per_session | book_resolved_rate | gold_retrieved_rate | citation_grounded_rate | gold_cited_rate | answer_support_mean | abstain_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 85 | 0.9059 [0.83, 0.95] | 0.975 | 5.69 | 42777 | 0.0 | 1.0 | 1.0 | 1.68 | 1.05 | 1.0 | 0.8462 | 1.0 | 1.0 | 0.4193 | 1.0 |
| kind=book_qa | 15 | 0.8667 [0.62, 0.96] | 1.0 | 4.33 | 29263 | None | None | None | 0.0 | 0.0 | 1.0 | 0.8462 | 1.0 | 1.0 | 0.4193 | 1.0 |
| kind=clarify | 4 | 0.75 [0.30, 0.95] | 1.0 | 2.25 | 8167 | None | None | None | 0.0 | 0.0 | None | None | None | None | None | None |
| kind=constrained_rec | 36 | 0.9444 [0.82, 0.98] | 0.9722 | 7.25 | 56900 | 0.0 | 1.0 | 1.0 | 1.92 | 1.0 | None | None | None | None | None | None |
| kind=memory | 16 | 0.9375 [0.72, 0.99] | 0.973 | 5.03 | 36385 | 0.0 | 1.0 | None | 1.92 | 0.78 | None | None | None | None | None | None |
| kind=session_state | 14 | 0.8571 [0.60, 0.96] | 0.9643 | 5.79 | 45250 | None | None | None | 2.18 | 2.18 | None | None | None | None | None | None |
| variant=accumulate | 5 | 1.0 [0.57, 1.00] | 1.0 | 3.8 | 23372 | 0.0 | 1.0 | None | 0.8 | 0.07 | None | None | None | None | None | None |
| variant=chapter_event | 3 | 1.0 [0.44, 1.00] | 1.0 | 3.67 | 21309 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.381 | None |
| variant=ending | 4 | 0.75 [0.30, 0.95] | 1.0 | 4.25 | 29366 | None | None | None | 0.0 | 0.0 | 1.0 | 0.75 | 1.0 | 1.0 | 0.4369 | None |
| variant=exclude | 5 | 0.6 [0.23, 0.88] | 0.9 | 7.2 | 58521 | None | None | None | 3.6 | 3.6 | None | None | None | None | None | None |
| variant=in_text | 12 | 0.9167 [0.65, 0.99] | 0.9167 | 7.83 | 62201 | 0.0 | 1.0 | None | 2.58 | 2.08 | None | None | None | None | None | None |
| variant=meta | 14 | 1.0 [0.78, 1.00] | 1.0 | 6.57 | 50065 | None | None | 1.0 | 0.0 | 0.0 | None | None | None | None | None | None |
| variant=mixed | 10 | 0.9 [0.60, 0.98] | 1.0 | 7.5 | 60109 | 0.0 | 1.0 | 1.0 | 3.8 | 1.1 | None | None | None | None | None | None |
| variant=more | 3 | 1.0 [0.44, 1.00] | 1.0 | 7.0 | 55324 | None | None | None | 3.17 | 3.17 | None | None | None | None | None | None |
| variant=oneoff | 8 | 1.0 [0.68, 1.00] | 1.0 | 5.62 | 42048 | 0.0 | 1.0 | None | 3.0 | 1.75 | None | None | None | None | None | None |
| variant=opening | 4 | 1.0 [0.51, 1.00] | 1.0 | 4.0 | 25926 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.4365 | None |
| variant=out_of_coverage | 2 | 1.0 [0.34, 1.00] | 1.0 | 5.0 | 34704 | None | None | None | 0.0 | 0.0 | 1.0 | None | None | None | None | 1.0 |
| variant=persist | 3 | 0.6667 [0.21, 0.94] | 0.8333 | 6.5 | 53816 | 0.0 | 1.0 | None | 1.83 | 0.0 | None | None | None | None | None | None |
| variant=quote_locate | 2 | 0.5 [0.09, 0.91] | 1.0 | 5.5 | 42221 | None | None | None | 0.0 | 0.0 | 1.0 | 0.5 | 1.0 | 1.0 | 0.4122 | None |
| variant=refer | 6 | 1.0 [0.61, 1.00] | 1.0 | 4.0 | 29152 | None | None | None | 0.5 | 0.5 | None | None | None | None | None | None |
| variant=vague | 4 | 0.75 [0.30, 0.95] | 1.0 | 2.25 | 8167 | None | None | None | 0.0 | 0.0 | None | None | None | None | None | None |

failure labels: no_recommendation=5, budget_exhausted=3, format=3, chapter_not_retrieved=2, no_clarification=1
