### Test set — 2,990 videos, one caption each

| Model                       |   P@5 |   R@5 |   F1@5 |   SetF1 (thr 0.5) |   SetF1 (val-tuned thr) |   #tags |   Category acc |
|-----------------------------|-------|-------|--------|-------------------|-------------------------|---------|----------------|
| Keyword rules (no learning) | 0.26  | 0.322 |  0.274 |             0.321 |                   0.321 |     1.5 |          0.141 |
| TF-IDF + LogReg             | 0.381 | 0.485 |  0.406 |             0.383 |                   0.402 |     2.3 |          0.491 |
| Transformer (BCE)           | 0.414 | 0.524 |  0.441 |             0.441 |                   0.461 |     2.4 |          0.517 |
| Transformer + RL (ours)     | 0.411 | 0.522 |  0.439 |             0.466 |                   0.465 |     3.4 |          0.515 |


### LLM comparison — fixed random subset of 200 test videos

| Model                              |   P@5 |   R@5 |   F1@5 |   SetF1 |   #tags |   Category acc |
|------------------------------------|-------|-------|--------|---------|---------|----------------|
| Keyword rules (no learning)        | 0.263 | 0.319 |  0.274 |   0.306 |     1.4 |          0.13  |
| TF-IDF + LogReg                    | 0.354 | 0.443 |  0.373 |   0.351 |     2.1 |          0.475 |
| Transformer (BCE)                  | 0.41  | 0.519 |  0.433 |   0.417 |     2.1 |          0.5   |
| Transformer + RL (ours)            | 0.405 | 0.511 |  0.428 |   0.448 |     3.1 |          0.495 |
| LLM zero-shot                      | 0.384 | 0.479 |  0.405 |   0.424 |     4.5 |          0.5   |
| Transformer+RL → LLM rerank (ours) | 0.428 | 0.53  |  0.451 |   0.451 |     5   |          0.515 |


### Efficiency — CPU, 1 thread

| Model                       |   Latency p50 (ms) |   p95 (ms) |   Throughput (captions/s, batch) |   Size (MB) | Params   |   Train (min, 2 vCPU) |
|-----------------------------|--------------------|------------|----------------------------------|-------------|----------|-----------------------|
| Keyword rules (no learning) |               0.14 |       0.29 |                            7,473 |         0   | –        |                   0   |
| TF-IDF + LogReg             |               0.61 |       0.72 |                           23,755 |        43.1 | 11.01M   |                   1.5 |
| Transformer (BCE)           |               1.83 |       2.55 |                            1,012 |        13.9 | 3.59M    |                  23.6 |
| Transformer + RL (ours)     |               1.84 |       2.02 |                            1,144 |        13.9 | 3.59M    |                  31.8 |


### Paired bootstrap (5,000 resamples)

| Comparison                                   |     Δ | 95% CI           | Significant   |
|----------------------------------------------|-------|------------------|---------------|
| RL vs BCE (SetF1, thr 0.5, full test)        | 0.025 | [+0.019, +0.030] | yes           |
| RL vs BCE+tuned threshold (SetF1, full test) | 0.005 | [+0.000, +0.009] | yes           |
| Hybrid vs RL (F1@5, LLM subset)              | 0.022 | [+0.001, +0.045] | yes           |
| Hybrid vs LLM zero-shot (F1@5, LLM subset)   | 0.027 | [+0.006, +0.047] | yes           |


### LLM cost per video

| Stage                              |   Input tokens |   Output tokens |   Calls |
|------------------------------------|----------------|-----------------|---------|
| LLM zero-shot                      |            899 |              16 |     200 |
| Transformer+RL → LLM rerank (ours) |            263 |              26 |     200 |
