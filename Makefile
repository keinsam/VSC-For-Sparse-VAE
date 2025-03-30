TESTS = 

###################################################################################
# PIPELINE

download:
	python3 -m script.download 

train:
	python3 -m script.train

graphics:
	echo TODO...

#########################################

test:
	echo TODO..

test_verbose:
	echo TODO..

train_just_vae:
	python3 -m script.train --model vae

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
	rm */__pycache__

clean_all: clean
	rm -rf data
	rm -rf output
	rm -rf cache