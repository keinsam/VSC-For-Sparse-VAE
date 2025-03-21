TESTS = 

test:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test);)

test_verbose:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test) -v;)

###################################################################################

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

clean:
	rm -rf web/_output web/.quarto web/src web/test
