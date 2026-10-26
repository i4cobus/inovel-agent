| group | n | pass_rate | finish_rate | mean_steps_per_session | mean_prompt_tokens_per_session | violation_rate_macro | check_term_coverage | check_trope_coverage | unrequested_checks_per_session | unrequested_offtopic_per_session | book_resolved_rate | gold_retrieved_rate | citation_grounded_rate | gold_cited_rate | answer_support_mean | abstain_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 85 | 0.8824 [0.80, 0.93] | 1.0 | 5.58 | 40419 | 0.0 | 0.9602 | 0.9653 | 1.82 | 0.95 | 1.0 | 0.9231 | 1.0 | 1.0 | 0.4149 | 1.0 |
| kind=book_qa | 15 | 0.9333 [0.70, 0.99] | 1.0 | 4.13 | 27881 | None | None | None | 0.0 | 0.0 | 1.0 | 0.9231 | 1.0 | 1.0 | 0.4149 | 1.0 |
| kind=clarify | 4 | 1.0 [0.51, 1.00] | 1.0 | 2.25 | 8043 | None | None | None | 0.0 | 0.0 | None | None | None | None | None | None |
| kind=constrained_rec | 36 | 0.8056 [0.65, 0.90] | 1.0 | 6.92 | 51707 | 0.0 | 0.9258 | 0.9653 | 2.81 | 0.39 | None | None | None | None | None | None |
| kind=memory | 16 | 1.0 [0.81, 1.00] | 1.0 | 5.11 | 35504 | 0.0 | 1.0 | None | 1.16 | 0.68 | None | None | None | None | None | None |
| kind=session_state | 14 | 0.8571 [0.60, 0.96] | 1.0 | 5.71 | 43745 | None | None | None | 2.68 | 2.68 | None | None | None | None | None | None |
| variant=accumulate | 5 | 1.0 [0.57, 1.00] | 1.0 | 4.4 | 26958 | 0.0 | 1.0 | None | 0.53 | 0.13 | None | None | None | None | None | None |
| variant=chapter_event | 3 | 1.0 [0.44, 1.00] | 1.0 | 3.67 | 20135 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.4405 | None |
| variant=ending | 4 | 1.0 [0.51, 1.00] | 1.0 | 4.0 | 28552 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3826 | None |
| variant=exclude | 5 | 0.6 [0.23, 0.88] | 1.0 | 6.8 | 50562 | None | None | None | 4.1 | 4.1 | None | None | None | None | None | None |
| variant=in_text | 12 | 0.9167 [0.65, 0.99] | 1.0 | 7.17 | 54046 | 0.0 | 0.9583 | None | 3.92 | 0.42 | None | None | None | None | None | None |
| variant=meta | 14 | 0.8571 [0.60, 0.96] | 1.0 | 6.14 | 45227 | None | None | 0.9405 | 0.21 | 0.0 | None | None | None | None | None | None |
| variant=mixed | 10 | 0.6 [0.31, 0.83] | 1.0 | 7.7 | 57970 | 0.0 | 0.8867 | 1.0 | 5.1 | 0.9 | None | None | None | None | None | None |
| variant=more | 3 | 1.0 [0.44, 1.00] | 1.0 | 6.17 | 51083 | None | None | None | 3.0 | 3.0 | None | None | None | None | None | None |
| variant=oneoff | 8 | 1.0 [0.68, 1.00] | 1.0 | 5.62 | 41584 | 0.0 | 1.0 | None | 1.69 | 1.19 | None | None | None | None | None | None |
| variant=opening | 4 | 1.0 [0.51, 1.00] | 1.0 | 4.5 | 30866 | None | None | None | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.4103 | None |
| variant=out_of_coverage | 2 | 1.0 [0.34, 1.00] | 1.0 | 4.5 | 29720 | None | None | None | 0.0 | 0.0 | 1.0 | None | 1.0 | None | None | 1.0 |
| variant=persist | 3 | 1.0 [0.44, 1.00] | 1.0 | 5.5 | 40654 | 0.0 | 1.0 | None | 1.33 | 0.67 | None | None | None | None | None | None |
| variant=quote_locate | 2 | 0.5 [0.09, 0.91] | 1.0 | 4.0 | 30346 | None | None | None | 0.0 | 0.0 | 1.0 | 0.5 | 1.0 | 1.0 | 0.4856 | None |
| variant=refer | 6 | 1.0 [0.61, 1.00] | 1.0 | 4.58 | 34394 | None | None | None | 1.33 | 1.33 | None | None | None | None | None | None |
| variant=vague | 4 | 1.0 [0.51, 1.00] | 1.0 | 2.25 | 8043 | None | None | None | 0.0 | 0.0 | None | None | None | None | None | None |

failure labels: tool_skipped=6, no_recommendation=5, chapter_not_retrieved=1
