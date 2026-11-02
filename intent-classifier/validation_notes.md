# Evaluation notes

The current classifier reached a macro F1 of **0.975** on the 40-example validation set and **0.948** on the separate 40-example test set in the verified November 1, 2026 CI run.

Next, focus on better data rather than optimizing repeatedly against these small sets. Useful additions include unfamiliar wording, multi-intent questions, and out-of-scope requests. Any future model comparison should preserve a genuinely untouched test set.
