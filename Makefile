.PHONY: data owt shakespeare

data: owt shakespeare

owt:
	uv run python data/openwebtext/prepare.py

shakespeare:
	uv run python data/shakespeare_char/prepare.py