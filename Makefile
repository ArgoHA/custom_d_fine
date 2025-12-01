.PHONY: full train preprocess split export bench infer test_batching check_errors ov_int8

full:
	python -m src.etl.preprocess
	python -m src.etl.split
	@$(MAKE) train
	python -m src.dl.export
	python -m src.dl.bench

preprocess:
	python -m src.etl.preprocess

split:
	python -m src.etl.split

train:
	@DDP_ENABLED=$$(python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('train', {}).get('ddp', {}).get('enabled', False))" 2>/dev/null || echo "False"); \
	if [ "$$DDP_ENABLED" = "True" ] || [ "$$DDP_ENABLED" = "true" ]; then \
		NUM_GPUS=$$(python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('train', {}).get('ddp', {}).get('n_gpus', 2))" 2>/dev/null || echo "2"); \
		echo "🚀 Training with DDP using $$NUM_GPUS GPUs..."; \
		torchrun --nproc_per_node=$$NUM_GPUS --master_port=29500 -m src.dl.train; \
	else \
		echo "🔧 Training with single GPU..."; \
		python -m src.dl.train; \
	fi

export:
	python -m src.dl.export

bench:
	python -m src.dl.bench

infer:
	python -m src.dl.infer

test_batching:
	python -m src.dl.test_batching

check_errors:
	python -m src.dl.check_errors

ov_int8:
	python -m src.dl.ov_int8
