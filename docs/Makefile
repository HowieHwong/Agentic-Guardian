# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Custom targets for AuraGen documentation

# Install documentation dependencies
install:
	pip install -r requirements.txt

# Build HTML documentation
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML documentation with live reload
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Clean build directory
clean:
	rm -rf "$(BUILDDIR)"
	@echo "Build directory cleaned."

# Build and open documentation in browser
serve: html
	@echo "Opening documentation in browser..."
	@python -c "import webbrowser; webbrowser.open('file://$(PWD)/$(BUILDDIR)/html/index.html')"

# Check for broken links
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Build PDF documentation (requires LaTeX)
latexpdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

# Build EPUB documentation
epub:
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

# Generate API documentation
apidoc:
	sphinx-apidoc -f -o "$(SOURCEDIR)/api" ../AuraGen
	@echo "API documentation generated."

# Full rebuild with API docs
rebuild: clean apidoc html
	@echo "Full rebuild completed."

# Check documentation for common issues
check:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Documentation checks completed."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
