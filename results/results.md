### Test set — 2,990 clips, one caption each (primary metric: wnDCG@5)

| Model                            |   wnDCG@5 |   nDCG@5 |   wRecall@5 | Hit@5   |   F1@5 |   SetF1 (0.5) |   SetF1 (val thr) | Cat top-1   | Cat top-2   |   Cat macro-F1 |
|----------------------------------|-----------|----------|-------------|---------|--------|---------------|-------------------|-------------|-------------|----------------|
| Keyword rules (no learning)      |     0.302 |    0.39  |       0.327 | 76.2%   |  0.274 |         0.321 |             0.321 | 14.1%       | 24.2%       |          0.067 |
| TF-IDF + LogReg                  |     0.501 |    0.531 |       0.547 | 83.4%   |  0.404 |         0.387 |             0.405 | 48.7%       | 62.0%       |          0.41  |
| Transformer from scratch (Keras) |     0.531 |    0.574 |       0.574 | 85.3%   |  0.439 |         0.447 |             0.461 | 52.1%       | 65.3%       |          0.436 |
| RoBERTa-base fine-tuned (BCE)    |     0.559 |    0.608 |       0.612 | 88.8%   |  0.465 |         0.472 |             0.489 | 54.7%       | 69.2%       |          0.458 |
| RoBERTa-base fine-tuned + RL     |     0.565 |    0.604 |       0.611 | 88.2%   |  0.46  |         0.489 |             0.491 | 54.6%       | 69.4%       |          0.444 |


### Clean test — 2,642 clips whose input caption never appears verbatim in training

| Model                            |   wnDCG@5 |   F1@5 | Hit@5   | Cat top-1   |
|----------------------------------|-----------|--------|---------|-------------|
| Keyword rules (no learning)      |     0.297 |  0.275 | 75.7%   | 13.7%       |
| TF-IDF + LogReg                  |     0.493 |  0.398 | 83.2%   | 48.0%       |
| Transformer from scratch (Keras) |     0.525 |  0.435 | 85.1%   | 51.9%       |
| RoBERTa-base fine-tuned (BCE)    |     0.558 |  0.465 | 89.2%   | 55.0%       |
| RoBERTa-base fine-tuned + RL     |     0.565 |  0.46  | 88.6%   | 54.8%       |


### LLM comparison — fixed random subset of 200 test clips

| Model                            |   wnDCG@5 |   nDCG@5 | Hit@5   |   F1@5 |   SetF1 | Cat top-1   |
|----------------------------------|-----------|----------|---------|--------|---------|-------------|
| Keyword rules (no learning)      |     0.298 |    0.386 | 79.0%   |  0.274 |   0.306 | 13.0%       |
| TF-IDF + LogReg                  |     0.485 |    0.5   | 82.0%   |  0.382 |   0.345 | 43.5%       |
| Transformer from scratch (Keras) |     0.523 |    0.555 | 84.5%   |  0.421 |   0.401 | 52.0%       |
| RoBERTa-base fine-tuned (BCE)    |     0.582 |    0.616 | 91.0%   |  0.461 |   0.474 | 58.5%       |
| RoBERTa-base fine-tuned + RL     |     0.592 |    0.613 | 91.5%   |  0.459 |   0.48  | 55.0%       |
| LLM zero-shot                    |     0.514 |    0.519 | 83.5%   |  0.405 |   0.424 | 50.0%       |
| RoBERTa+RL → LLM rerank          |     0.543 |    0.614 | 89.0%   |  0.466 |   0.466 | 57.0%       |


### Paired bootstrap (5,000 resamples)

| Comparison                                                            |      Δ | 95% CI           | Significant   |
|-----------------------------------------------------------------------|--------|------------------|---------------|
| RoBERTa+RL vs Transformer from scratch (Keras) (wnDCG@5)              |  0.034 | [+0.026, +0.043] | yes           |
| RoBERTa+RL vs TF-IDF + LogReg (wnDCG@5)                               |  0.064 | [+0.055, +0.073] | yes           |
| RoBERTa+RL vs Keyword rules (no learning) (wnDCG@5)                   |  0.263 | [+0.252, +0.274] | yes           |
| RoBERTa-BCE vs Transformer from scratch (wnDCG@5)                     |  0.028 | [+0.019, +0.036] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (Set-F1, thr 0.5)                           |  0.017 | [+0.011, +0.022] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (wnDCG@5)                                   |  0.007 | [+0.003, +0.011] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (F1@5, fixed 5 tags)                        | -0.004 | [-0.008, -0.001] | yes           |
| RoBERTa+RL (thr 0.5) vs RoBERTa-BCE with val-tuned threshold (Set-F1) | -0     | [-0.005, +0.005] | no            |
| LLM rerank vs RoBERTa+RL (wnDCG@5, subset)                            | -0.049 | [-0.076, -0.023] | yes           |
| LLM rerank vs RoBERTa+RL (F1@5, subset)                               |  0.006 | [-0.013, +0.026] | no            |
| LLM zero-shot vs RoBERTa+RL (wnDCG@5, subset)                         | -0.077 | [-0.120, -0.033] | yes           |


### Efficiency — CPU, 1 thread

| Model                            |   Latency p50 (ms) |   p95 (ms) |   Throughput (captions/s) |   Size (MB) | Params   |   Train (min, 2 vCPU) |
|----------------------------------|--------------------|------------|---------------------------|-------------|----------|-----------------------|
| Keyword rules (no learning)      |               0.18 |       0.36 |                     5,881 |           0 | –        |                   0   |
| TF-IDF + LogReg                  |               1.18 |       1.44 |                    34,032 |          43 | 11.0M    |                   0.9 |
| Transformer from scratch (Keras) |               2.3  |       3.09 |                     1,239 |          14 | 3.6M     |                  23.8 |
| RoBERTa-base fine-tuned (BCE) †  |             112.87 |     132.04 |                        32 |         474 | 124.3M   |                  78.5 |
| RoBERTa-base fine-tuned + RL †   |             112.87 |     132.04 |                        32 |         474 | 124.3M   |                  89.5 |

† measured in an earlier run of this pipeline; the checkpoint is not kept on disk (size / not committed).


### LLM cost per clip (tokens estimated at ~4 chars/token)

| Stage                   |   Input tokens |   Output tokens |   Calls |
|-------------------------|----------------|-----------------|---------|
| LLM zero-shot           |            899 |              16 |     200 |
| RoBERTa+RL → LLM rerank |            263 |              26 |     200 |
