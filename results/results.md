### Test set — 2,990 clips, one caption each (primary metric: wnDCG@5)

| Model                          |   wnDCG@5 |   nDCG@5 |   wRecall@5 | Hit@5   |   F1@5 |   SetF1 (0.5) |   SetF1 (val thr) | Cat top-1   | Cat top-2   |   Cat macro-F1 |
|--------------------------------|-----------|----------|-------------|---------|--------|---------------|-------------------|-------------|-------------|----------------|
| Keyword rules (no learning)    |     0.304 |    0.392 |       0.334 | 76.2%   |  0.275 |         0.315 |             0.315 | 14.0%       | 24.0%       |          0.065 |
| TF-IDF + LogReg                |     0.499 |    0.53  |       0.546 | 83.8%   |  0.404 |         0.38  |             0.398 | 49.1%       | 62.1%       |          0.412 |
| Transformer from scratch (BCE) |     0.536 |    0.581 |       0.582 | 86.0%   |  0.441 |         0.444 |             0.462 | 52.5%       | 66.1%       |          0.434 |
| Transformer from scratch + RL  |     0.552 |    0.577 |       0.589 | 85.8%   |  0.438 |         0.468 |             0.468 | 52.0%       | 65.9%       |          0.419 |
| RoBERTa-base fine-tuned (BCE)  |     0.552 |    0.605 |       0.604 | 88.3%   |  0.459 |         0.471 |             0.486 | 54.7%       | 69.3%       |          0.455 |
| RoBERTa-base fine-tuned + RL   |     0.555 |    0.597 |       0.599 | 87.2%   |  0.453 |         0.485 |             0.49  | 53.9%       | 68.3%       |          0.433 |


### Clean test — 2,642 clips whose input caption never appears verbatim in training

| Model                          |   wnDCG@5 |   F1@5 | Hit@5   | Cat top-1   |
|--------------------------------|-----------|--------|---------|-------------|
| Keyword rules (no learning)    |     0.3   |  0.277 | 76.0%   | 13.7%       |
| TF-IDF + LogReg                |     0.492 |  0.397 | 83.6%   | 48.5%       |
| Transformer from scratch (BCE) |     0.531 |  0.438 | 86.0%   | 52.3%       |
| Transformer from scratch + RL  |     0.549 |  0.435 | 85.8%   | 51.8%       |
| RoBERTa-base fine-tuned (BCE)  |     0.552 |  0.459 | 88.8%   | 55.0%       |
| RoBERTa-base fine-tuned + RL   |     0.556 |  0.454 | 87.6%   | 54.0%       |


### LLM comparison — fixed random subset of 200 test clips

| Model                          |   wnDCG@5 |   nDCG@5 | Hit@5   |   F1@5 |   SetF1 | Cat top-1   |
|--------------------------------|-----------|----------|---------|--------|---------|-------------|
| Keyword rules (no learning)    |     0.3   |    0.386 | 77.0%   |  0.277 |   0.302 | 13.0%       |
| TF-IDF + LogReg                |     0.484 |    0.499 | 82.0%   |  0.375 |   0.345 | 47.5%       |
| Transformer from scratch (BCE) |     0.546 |    0.577 | 86.5%   |  0.431 |   0.433 | 52.5%       |
| Transformer from scratch + RL  |     0.563 |    0.576 | 85.5%   |  0.428 |   0.453 | 52.5%       |
| RoBERTa-base fine-tuned (BCE)  |     0.57  |    0.606 | 89.5%   |  0.449 |   0.467 | 56.5%       |
| RoBERTa-base fine-tuned + RL   |     0.571 |    0.594 | 88.5%   |  0.439 |   0.466 | 55.5%       |
| LLM zero-shot                  |     0.511 |    0.519 | 82.0%   |  0.401 |   0.411 | 50.0%       |
| Transformer+RL → LLM rerank    |     0.548 |    0.56  | 84.5%   |  0.441 |   0.441 | 52.5%       |


### Paired bootstrap (5,000 resamples)

| Comparison                                     |      Δ | 95% CI           | Significant   |
|------------------------------------------------|--------|------------------|---------------|
| RoBERTa-BCE vs Transformer-BCE (wnDCG@5)       |  0.016 | [+0.009, +0.024] | yes           |
| RoBERTa+RL vs Transformer+RL (wnDCG@5)         |  0.003 | [-0.005, +0.010] | no            |
| RoBERTa+RL vs Transformer+RL (Set-F1)          |  0.017 | [+0.010, +0.024] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (Set-F1)             |  0.014 | [+0.008, +0.019] | yes           |
| RoBERTa+RL vs RoBERTa-BCE (wnDCG@5)            |  0.003 | [-0.001, +0.007] | no            |
| Transformer+RL vs Transformer-BCE (Set-F1)     |  0.024 | [+0.019, +0.030] | yes           |
| LLM rerank vs Transformer+RL (wnDCG@5, subset) | -0.014 | [-0.053, +0.024] | no            |
| LLM rerank vs Transformer+RL (F1@5, subset)    |  0.013 | [-0.010, +0.036] | no            |
| RoBERTa+RL vs LLM rerank (wnDCG@5, subset)     |  0.023 | [-0.015, +0.062] | no            |


### Efficiency — CPU, 1 thread

| Model                          |   Latency p50 (ms) |   p95 (ms) |   Throughput (captions/s) |   Size (MB) | Params   |   Train (min, 2 vCPU) |
|--------------------------------|--------------------|------------|---------------------------|-------------|----------|-----------------------|
| Keyword rules (no learning)    |               6.86 |      11.7  |                    29,045 |           0 | –        |                   0   |
| TF-IDF + LogReg                |               0.48 |       0.57 |                    27,645 |          45 | 11.6M    |                   1.4 |
| Transformer from scratch (BCE) |               1.31 |       1.58 |                     1,579 |          14 | 3.6M     |                  24   |
| Transformer from scratch + RL  |               1.3  |       1.47 |                     1,799 |          14 | 3.6M     |                  31.9 |
| RoBERTa-base fine-tuned (BCE)  |              77.39 |      98.88 |                        46 |         474 | 124.4M   |                  60.6 |
| RoBERTa-base fine-tuned + RL   |              77.39 |      98.88 |                        46 |         474 | 124.4M   |                  68.7 |


### LLM cost per clip (tokens estimated at ~4 chars/token)

| Stage                       |   Input tokens |   Output tokens |   Calls |
|-----------------------------|----------------|-----------------|---------|
| LLM zero-shot               |            921 |              16 |     200 |
| Transformer+RL → LLM rerank |            262 |              26 |     200 |
