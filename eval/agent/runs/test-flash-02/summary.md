| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session | unrequested_offtopic_per_session | book_resolved_rate | gold_retrieved_rate | citation_grounded_rate | gold_cited_rate | answer_support_mean | abstain_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 307 | 0.8404 [0.80, 0.88] | 0.9932 | 5.81 | 43031 | 0.0 | 0.9413 | 0.9282 | 1.97 | 0.96 | 1.0 | 0.9714 | 1.0 | 0.9706 | 0.4095 | 1.0 |
| kind=book_qa | 45 | 0.9333 [0.82, 0.98] | 1.0 | 4.47 | 33033 | None | None | None | 0.0 | 0.0 | 1.0 | 0.9714 | 1.0 | 0.9706 | 0.4095 | 1.0 |
| kind=clarify | 12 | 0.8333 [0.55, 0.95] | 1.0 | 2.75 | 11284 | None | None | None | 0.25 | 0.25 | None | None | None | None | None | None |
| kind=constrained_rec | 140 | 0.8143 [0.74, 0.87] | 0.9857 | 7.26 | 55191 | 0.0 | 0.9651 | 0.9282 | 2.6 | 1.02 | None | None | None | None | None | None |
| kind=memory | 60 | 0.7667 [0.65, 0.86] | 0.9929 | 5.02 | 34585 | 0.0 | 0.9126 | None | 2.32 | 0.76 | None | None | None | None | None | None |
| kind=session_state | 50 | 0.92 [0.81, 0.97] | 1.0 | 5.87 | 46223 | None | None | None | 1.69 | 1.69 | None | None | None | None | None | None |
| variant=accumulate | 21 | 0.619 [0.41, 0.79] | 0.9841 | 4.02 | 23964 | 0.0 | 0.8083 | None | 0.68 | 0.16 | None | None | None | None | None | None |
| variant=chapter_event | 9 | 0.8889 [0.56, 0.98] | 1.0 | 3.89 | 26353 | None | None | None | 0.0 | 0.0 | 1.0 | 0.8889 | 1.0 | 1.0 | 0.4479 | None |
| variant=ending | 8 | 1.0 [0.68, 1.00] | 1.0 | 4.25 | 29063 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3926 | None |
| variant=exclude | 16 | 0.875 [0.64, 0.96] | 1.0 | 6.59 | 51476 | None | None | None | 1.91 | 1.91 | None | None | None | None | None | None |
| variant=in_text | 50 | 0.86 [0.74, 0.93] | 1.0 | 7.36 | 56898 | 0.0 | 0.995 | None | 4.56 | 1.5 | None | None | None | None | None | None |
| variant=meta | 40 | 0.8 [0.65, 0.90] | 1.0 | 7.05 | 53617 | None | None | 0.9423 | 0.45 | 0.28 | None | None | None | None | None | None |
| variant=mixed | 50 | 0.78 [0.65, 0.87] | 0.96 | 7.32 | 54744 | 0.0 | 0.934 | 0.9167 | 2.36 | 1.14 | None | None | None | None | None | None |
| variant=more | 19 | 0.8947 [0.69, 0.97] | 1.0 | 6.47 | 53214 | None | None | None | 2.32 | 2.32 | None | None | None | None | None | None |
| variant=oneoff | 16 | 0.875 [0.64, 0.96] | 1.0 | 6.12 | 45561 | 0.0 | 0.9167 | None | 4.12 | 1.72 | None | None | None | None | None | None |
| variant=opening | 8 | 1.0 [0.68, 1.00] | 1.0 | 4.5 | 29043 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.4282 | None |
| variant=out_of_coverage | 10 | 1.0 [0.72, 1.00] | 1.0 | 4.3 | 30089 | None | None | None | 0.0 | 0.0 | 1.0 | None | 1.0 | None | None | 1.0 |
| variant=persist | 23 | 0.8261 [0.63, 0.93] | 1.0 | 5.63 | 41496 | 0.0 | 0.9574 | None | 3.3 | 0.91 | None | None | None | None | None | None |
| variant=quote_locate | 10 | 0.8 [0.49, 0.94] | 1.0 | 5.3 | 48358 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.9 | 0.3775 | None |
| variant=refer | 15 | 1.0 [0.80, 1.00] | 1.0 | 4.33 | 31763 | None | None | None | 0.67 | 0.67 | None | None | None | None | None | None |
| variant=vague | 12 | 0.8333 [0.55, 0.95] | 1.0 | 2.75 | 11284 | None | None | None | 0.25 | 0.25 | None | None | None | None | None | None |

failure labels: tool_skipped=25, no_recommendation=20, format=3, budget_exhausted=2, recommended_without_asking=2, answer_unsupported=2, no_clarification=1, chapter_not_retrieved=1, gold_not_cited=1
