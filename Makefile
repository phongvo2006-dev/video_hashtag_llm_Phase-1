PY ?= python

.PHONY: all data baselines transformer roberta llm eval figures test demo clean

all: data baselines transformer roberta llm eval figures

data:            ## download MSR-VTT annotations + build hashtag labels
	bash scripts/download_data.sh
	$(PY) scripts/01_prepare_data.py

baselines:       ## keyword rules + TF-IDF logistic regression (TensorFlow)
	$(PY) scripts/02_train_baselines.py

transformer:     ## Transformer encoder from scratch (TensorFlow/Keras, BCE)
	$(PY) scripts/03_train_transformer.py

roberta:         ## fine-tune pretrained RoBERTa-base (BCE, then RL); add RESUME=1 to continue an interrupted run
	$(PY) scripts/04_finetune_roberta.py $(if $(RESUME),--resume,)

llm:             ## LLM zero-shot + hybrid rerank of RoBERTa+RL candidates (uses cache unless provider is set)
	$(PY) scripts/05_llm_hashtags.py --stage zero_shot
	$(PY) scripts/05_llm_hashtags.py --stage hybrid

eval:            ## metrics, significance tests, efficiency
	$(PY) scripts/06_evaluate.py

figures:
	$(PY) scripts/07_make_figures.py

test:
	$(PY) -m pytest -q tests

demo:
	$(PY) demo/app.py

clean:
	rm -rf data/processed results/preds checkpoints/*.pkl
