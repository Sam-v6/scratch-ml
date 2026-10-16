# Timestral comparison plan

Goal: runnable local next-step comparison and inspectable plots on the committed synthetic sensor data.

1. Add an optional, commit-pinned Timestral dependency and a small adapter accepting (batch, time, four channels). Target history is channel four; sensors are past-only covariates. Verify axis mapping and absence of future inputs with a recording fake, then run real weights.
2. Add a standalone benchmark reusing all four existing model classes with four input channels. Use 256 past samples, one-step horizon, chronological 60/20/20 split, train-only scaling, validation-only checkpoint selection, fixed seed and 50 epochs. Test window boundaries and test-future isolation before implementation.
3. Compare persistence, a 100-step seasonal baseline, four trained models, and zero-shot Timestral 5M/28M/209M. Save predictions, metrics, training curves, pinned model revisions, data hash, and plots. Verify finite predictions and recompute saved metrics.
4. Document commands, actual results, timing scope, and limits. Inspect plots and run unit tests/lint before completion.

Scope: CUDA on RTX 4090 (user requested GPU after initial sandbox check), no changes to existing training/export pipeline, no claim of multi-step or real-world generalization. Rolling evaluation observes earlier test targets only after they occur. Existing data's full-series rescaling is disclosed. Main checkout edits remain untouched.
