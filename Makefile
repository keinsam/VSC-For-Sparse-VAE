TESTS = 

###################################################################################
# PIPELINE

pip:
	pip install -r requirements.txt

download:
	python3 -m script.download 

train:
	python3 -m script.train

graphics:
	python3 -m script.graphics

#########################################

test:
	echo TODO..

test_verbose:
	echo TODO..

###################################################################################
# REPORT

render:
	mkdir -p web/_output src test
	cp -ru src web
	cp -ru test web
	cd web; quarto render

#########################################

update_web: render
	rm -rf docs
	mkdir -p docs
	cp -r web/_output/* docs

preview_web:
	mkdir -p web/_output src test
	cp -ru src web
	cp -ru test web
	cd web; quarto preview

preview_pdf: render
	xdg-open web/_output/*.pdf

###################################################################################
# MISC

clean:
	rm -rf web/_output web/.quarto web/src web/test
	rm -rf */__pycache__ */*/__pycache__

clean_data: # remove datasets
	rm -rf data

clean_cache: # remove stored weights
	rm -rf cache

clean_output:
	rm -rf output

clean_all: clean clean_cache clean_output clean_data

###################################################################################

all: 
	@echo "Specify a target. Default behavior is disabled."