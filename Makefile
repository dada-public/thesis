.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

include .env

PYTHON_INTERPRETER = python3

$(V).SILENT:

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Cleans all generated data in this project
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	$(PYTHON_INTERPRETER) src/data/make_clean.py

## Gathers GTZAN dataset and pre-process its contents.
data: clean
	$(PYTHON_INTERPRETER) src/data/make_data.py

## Creates a featured TRAIN/TEST stratified split
## rs=(42), use={SPEC|MEL|CHROMA}
features:
	$(PYTHON_INTERPRETER) src/data/make_features.py rs=$(rs) use=$(use) dim=$(dim)

## Creates features and trains a model.
## rs=(42), use={SPEC|MEL|CHROMA} dim={1|2}
train: features
	$(PYTHON_INTERPRETER) src/data/make_train.py rs=$(rs) use=$(use) dim=$(dim)

## Eval: trains and evaluates a series of models.
## rs=(42), use={SPEC|MEL|CHROMA} dim={1|2} batch=(100)
eval:
	n=$(batch); \
	while [ $${n} -gt 0 ] ; do \
		$(PYTHON_INTERPRETER) src/data/make_features.py rs=$${rs} use=$(use) dim=$(dim); \
		$(PYTHON_INTERPRETER) src/data/make_train.py rs=$${rs} use=$(use) dim=$(dim); \
		$(PYTHON_INTERPRETER) src/data/make_eval.py rs=$${rs} use=$(use) dim=$(dim); \
		n=`expr $$n - 1`; \
		rs=`expr $$rs + 1`; \
	done; \
	true

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
