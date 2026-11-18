PY ?= python

.PHONY: all data baselines transformer rl roberta llm eval figures test demo clean

all: data baselines transformer rl roberta llm eval figures

data:            ## download MSR-VTT annotations + build hashtag labels
	bash scripts/download_data.sh
	$(PY) scripts/01_prepare_data.py

baselines:       ## keyword rules + TF-IDF logistic regression
	$(PY) scripts/02_train_baselines.py

transformer:     ## supervised (BCE) Transformer
	$(PY) scripts/03_train_transformer.py

rl:              ## REINFORCE/RLOO fine-tuning on set-F1 reward
	$(PY) scripts/04_rl_finetune.py

roberta:         ## fine-tune pretrained RoBERTa-base (BCE + RL); add RESUME=1 to continue an interrupted run
	$(PY) scripts/08_finetune_roberta.py $(if $(RESUME),--resume,)

llm:             ## LLM zero-shot + hybrid rerank (uses cache unless provider is set)
	$(PY) scripts/05_llm_hashtags.py --stage zero_shot
	$(PY) scripts/05_llm_hashtags.py --stage hybrid

eval:
	$(PY) scripts/06_evaluate.py

figures:
	$(PY) scripts/07_make_figures.py

test:
	$(PY) -m pytest -q tests

demo:
	$(PY) demo/app.py

clean:
	rm -rf data/processed results/preds checkpoints/*.pkl
