### Test set — 2,990 clips, one caption each (primary metric: wnDCG@5)

| Model                          |   wnDCG@5 |   nDCG@5 |   wRecall@5 | Hit@5   |   F1@5 |   SetF1 (0.5) |   SetF1 (val thr) | Cat top-1   | Cat top-2   |   Cat macro-F1 |
|--------------------------------|-----------|----------|-------------|---------|--------|---------------|-------------------|-------------|-------------|----------------|
| Keyword rules (no learning)    |     0.302 |    0.39  |       0.327 | 76.2%   |  0.274 |         0.321 |             0.321 | 14.1%       | 24.2%       |          0.067 |
| TF-IDF + LogReg                |     0.501 |    0.531 |       0.55  | 84.2%   |  0.406 |         0.383 |             0.402 | 49.1%       | 62.1%       |          0.412 |
| Transformer from scratch (BCE) |     0.537 |    0.579 |       0.584 | 86.3%   |  0.441 |         0.441 |             0.461 | 51.7%       | 65.5%       |          0.424 |
| Transformer from scratch + RL  |     0.551 |    0.573 |       0.591 | 85.7%   |  0.439 |         0.466 |             0.465 | 51.5%       | 65.3%       |          0.416 |
| Keras Transformer (BCE)        |     0.531 |    0.574 |       0.574 | 85.3%   |  0.439 |         0.447 |             0.461 | 52.1%       | 65.3%       |          0.436 |
| Keras Transformer + RL         |     0.547 |    0.579 |       0.586 | 85.7%   |  0.444 |         0.472 |             0.473 | 52.6%       | 65.1%       |          0.436 |
| RoBERTa-base fine-tuned (BCE)  |     0.559 |    0.608 |       0.612 | 88.8%   |  0.465 |         0.472 |             0.489 | 54.7%       | 69.2%       |          0.458 |
| RoBERTa-base fine-tuned + RL   |     0.565 |    0.604 |       0.611 | 88.2%   |  0.46  |         0.489 |             0.491 | 54.6%       | 69.4%       |          0.444 |


### Clean test — 2,642 clips whose input caption never appears verbatim in training

| Model                          |   wnDCG@5 |   F1@5 | Hit@5   | Cat top-1   |
|--------------------------------|-----------|--------|---------|-------------|
| Keyword rules (no learning)    |     0.297 |  0.275 | 75.7%   | 13.7%       |
| TF-IDF + LogReg                |     0.493 |  0.399 | 84.0%   | 48.5%       |
| Transformer from scratch (BCE) |     0.53  |  0.437 | 86.2%   | 51.4%       |
| Transformer from scratch + RL  |     0.548 |  0.436 | 85.7%   | 51.2%       |
| Keras Transformer (BCE)        |     0.525 |  0.435 | 85.1%   | 51.9%       |
| Keras Transformer + RL         |     0.543 |  0.441 | 85.7%   | 52.3%       |
| RoBERTa-base fine-tuned (BCE)  |     0.558 |  0.465 | 89.2%   | 55.0%       |
| RoBERTa-base fine-tuned + RL   |     0.565 |  0.46  | 88.6%   | 54.8%       |


### LLM comparison — fixed random subset of 200 test clips

| Model                          |   wnDCG@5 |   nDCG@5 | Hit@5   |   F1@5 |   SetF1 | Cat top-1   |
|--------------------------------|-----------|----------|---------|--------|---------|-------------|
| Keyword rules (no learning)    |     0.298 |    0.386 | 79.0%   |  0.274 |   0.306 | 13.0%       |
| TF-IDF + LogReg                |     0.483 |    0.499 | 83.0%   |  0.373 |   0.351 | 47.5%       |
| Transformer from scratch (BCE) |     0.545 |    0.578 | 86.0%   |  0.433 |   0.417 | 50.0%       |
| Transformer from scratch + RL  |     0.556 |    0.572 | 86.0%   |  0.428 |   0.448 | 49.5%       |
| Keras Transformer (BCE)        |     0.523 |    0.555 | 84.5%   |  0.421 |   0.401 | 52.0%       |
| Keras Transformer + RL         |     0.541 |    0.562 | 83.5%   |  0.426 |   0.457 | 54.5%       |
| RoBERTa-base fine-tuned (BCE)  |     0.582 |    0.616 | 91.0%   |  0.461 |   0.474 | 58.5%       |
| RoBERTa-base fine-tuned + RL   |     0.592 |    0.613 | 91.5%   |  0.459 |   0.48  | 55.0%       |
| LLM zero-shot                  |     0.514 |    0.519 | 83.5%   |  0.405 |   0.424 | 50.0%       |
| Transformer+RL → LLM rerank    |     0.546 |    0.563 | 85.5%   |  0.451 |   0.451 | 51.5%       |


### Paired bootstrap (5,000 resamples)

| Comparison                                                |      Δ | 95% CI           | Significant   |
|-----------------------------------------------------------|--------|------------------|---------------|
| RoBERTa-BCE vs Transformer-BCE (wnDCG@5)                  |  0.022 | [+0.014, +0.029] | yes           |
| RoBERTa+RL vs Transformer+RL (wnDCG@5)                    |  0.014 | [+0.007, +0.021] | yes           |
| RoBERTa+RL vs Transformer+RL (Set-F1)                     |  0.023 | [+0.017, +0.030] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (Set-F1)                        |  0.017 | [+0.011, +0.022] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (wnDCG@5)                       |  0.007 | [+0.003, +0.011] | yes           |
| Transformer+RL vs Transformer-BCE (Set-F1)                |  0.025 | [+0.019, +0.030] | yes           |
| Keras+RL vs Keras-BCE (Set-F1)                            |  0.025 | [+0.020, +0.030] | yes           |
| Keras (TensorFlow)+RL vs PyTorch Transformer+RL (wnDCG@5) | -0.004 | [-0.011, +0.004] | no            |
| Keras (TensorFlow)+RL vs PyTorch Transformer+RL (Set-F1)  |  0.006 | [-0.000, +0.013] | no            |
| LLM rerank vs Transformer+RL (wnDCG@5, subset)            | -0.01  | [-0.047, +0.027] | no            |
| LLM rerank vs Transformer+RL (F1@5, subset)               |  0.022 | [+0.001, +0.045] | yes           |
| RoBERTa+RL vs LLM rerank (wnDCG@5, subset)                |  0.046 | [+0.007, +0.085] | yes           |


### Efficiency — CPU, 1 thread

| Model                            |   Latency p50 (ms) |   p95 (ms) |   Throughput (captions/s) |   Size (MB) | Params   |   Train (min, 2 vCPU) |
|----------------------------------|--------------------|------------|---------------------------|-------------|----------|-----------------------|
| Keyword rules (no learning)      |               0.18 |       0.32 |                       886 |           0 | –        |                   0   |
| TF-IDF + LogReg                  |               0.64 |       0.81 |                    21,105 |          43 | 11.0M    |                   2   |
| Transformer from scratch (BCE) † |               2.37 |       2.8  |                     1,121 |          14 | 3.6M     |                  23.6 |
| Transformer from scratch + RL    |               1.91 |       2.09 |                     1,234 |          14 | 3.6M     |                  31.8 |
| Keras Transformer (BCE)          |               2.26 |       2.96 |                     1,247 |          14 | 3.6M     |                  23.8 |
| Keras Transformer + RL           |               2.26 |       2.96 |                     1,247 |          14 | 3.6M     |                  31.9 |
| RoBERTa-base fine-tuned (BCE) †  |             112.87 |     132.04 |                        32 |         474 | 124.3M   |                  78.5 |
| RoBERTa-base fine-tuned + RL †   |             112.87 |     132.04 |                        32 |         474 | 124.3M   |                  89.5 |

† measured in an earlier run of this pipeline; the checkpoint is not kept on disk (size / not committed).


### LLM cost per clip (tokens estimated at ~4 chars/token)

| Stage                       |   Input tokens |   Output tokens |   Calls |
|-----------------------------|----------------|-----------------|---------|
| LLM zero-shot               |            899 |              16 |     200 |
| Transformer+RL → LLM rerank |            263 |              26 |     200 |
