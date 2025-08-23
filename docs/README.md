# AuraGen Documentation

This directory contains the complete documentation for AuraGen, built with [Sphinx](https://www.sphinx-doc.org/) and styled with the [Read the Docs theme](https://sphinx-rtd-theme.readthedocs.io/).

## 📚 Documentation Structure

```
docs/
├── source/                     # Documentation source files
│   ├── _static/               # Static assets (CSS, images)
│   ├── _templates/            # Custom templates
│   ├── api/                   # API reference documentation
│   ├── advanced/              # Advanced topics
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation page
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst        # Quick start guide
│   ├── configuration.rst     # Configuration documentation
│   ├── scenarios.rst         # Scenarios guide
│   └── risk_injection.rst    # Risk injection guide
├── build/                     # Generated documentation (auto-created)
├── requirements.txt          # Documentation dependencies
├── Makefile                  # Build commands
├── build_docs.sh            # Quick build script
└── README.md                # This file
```

## 🚀 Quick Start

### Build Documentation

The easiest way to build the documentation:

```bash
# Navigate to docs directory
cd docs/

# Run the build script
./build_docs.sh
```

This script will:
- Install all required dependencies
- Clean previous builds
- Generate HTML documentation
- Provide instructions for viewing

### Manual Build

If you prefer manual control:

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML documentation
make html

# Serve locally
cd build/html
python -m http.server 8000
# Visit http://localhost:8000
```

## 📖 Documentation Features

### Comprehensive Coverage

- **Installation Guide**: Complete setup instructions
- **Quick Start**: Get running in minutes
- **Configuration**: Detailed configuration options
- **API Reference**: Full API documentation with examples
- **Advanced Topics**: In-depth guides for power users

### Interactive Elements

- **Code Examples**: Copy-pastable code snippets
- **Configuration Examples**: Real-world configuration files
- **Troubleshooting**: Common issues and solutions
- **Cross-references**: Easy navigation between topics

### Professional Styling

- **Read the Docs Theme**: Clean, professional appearance
- **Custom CSS**: Enhanced styling for better readability
- **Responsive Design**: Works on all devices
- **Syntax Highlighting**: Beautiful code formatting

## 🛠️ Development

### Adding New Documentation

1. **Create new `.rst` files** in `source/` directory
2. **Add to table of contents** in `index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2
      
      new_page
   ```
3. **Build and test** with `make html`

### Updating API Documentation

API documentation is automatically generated from docstrings:

```bash
# Regenerate API docs
make apidoc

# Full rebuild
make rebuild
```

### Documentation Standards

- **Use reStructuredText** (`.rst`) format
- **Include code examples** for all features
- **Add cross-references** with `:doc:` role
- **Write clear headings** with proper hierarchy
- **Test all examples** before committing

## 📝 Writing Guidelines

### Structure

```rst
Page Title
==========

Overview section introducing the topic.

Section Heading
---------------

Content with examples.

Subsection Heading
~~~~~~~~~~~~~~~~~~

More detailed content.

.. code-block:: python

   # Example code
   from AuraGen import core
```

### Best Practices

- **Start with overview**: Brief introduction to the topic
- **Use examples**: Show, don't just tell
- **Cross-reference**: Link to related sections
- **Be consistent**: Follow established patterns
- **Test thoroughly**: Ensure all examples work

### Code Examples

Always include working examples:

```rst
.. code-block:: python

   from AuraGen.core import AuraGenCore
   
   # Initialize core
   core = AuraGenCore()
   
   # Generate trajectories
   trajectories = core.generate_trajectories(
       scenario_name="email_assistant",
       num_records=10
   )
```

## 🔧 Available Commands

### Makefile Commands

```bash
make html          # Build HTML documentation
make clean         # Clean build directory  
make livehtml      # Build with live reload
make linkcheck     # Check for broken links
make serve         # Build and open in browser
make apidoc        # Generate API documentation
make rebuild       # Full clean rebuild
```

### Manual Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML
sphinx-build -b html source build/html

# Build with live reload
sphinx-autobuild source build/html

# Check links
sphinx-build -b linkcheck source build/linkcheck
```

## 🎨 Customization

### Theme Customization

Edit `source/_static/custom.css` to customize appearance:

```css
/* Custom styles */
.wy-nav-top {
    background: linear-gradient(45deg, #2980b9, #3498db);
}
```

### Configuration

Edit `source/conf.py` for Sphinx configuration:

```python
# Project information
project = 'AuraGen'
author = 'AuraGen Team'

# Theme options
html_theme_options = {
    'collapse_navigation': True,
    'navigation_depth': 4,
}
```

## 📋 Dependencies

### Core Dependencies

- `sphinx>=4.0.0` - Documentation generator
- `sphinx-rtd-theme>=1.0.0` - Read the Docs theme
- `myst-parser>=0.18.0` - Markdown support

### Optional Dependencies

- `sphinx-autodoc-typehints` - Enhanced type hints
- `sphinx-copybutton` - Copy code button
- `sphinx-tabs` - Tabbed content
- `sphinxcontrib-mermaid` - Diagram support

## 🔍 Troubleshooting

### Common Issues

**Build Failures**

```bash
# Clean and rebuild
make clean
make html
```

**Missing Dependencies**

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Import Errors**

Check that AuraGen is properly installed:

```bash
# Install in development mode
pip install -e .
```

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review [reStructuredText guide](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- Ask questions in project issues

## 🚀 Deployment

### GitHub Pages

Add to `.github/workflows/docs.yml`:

```yaml
name: Build and Deploy Docs

on:
  push:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        cd docs
        pip install -r requirements.txt
    - name: Build docs
      run: |
        cd docs
        make html
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/build/html
```

### Read the Docs

1. Connect repository to [Read the Docs](https://readthedocs.org/)
2. Configure build settings:
   - **Requirements file**: `docs/requirements.txt`
   - **Python version**: 3.8+
   - **Sphinx configuration**: `docs/source/conf.py`

## 🤝 Contributing

1. **Fork the repository**
2. **Create documentation branch**: `git checkout -b docs/feature-name`
3. **Make changes** and test locally
4. **Submit pull request** with clear description

### Review Checklist

- [ ] Documentation builds without errors
- [ ] All code examples work
- [ ] Cross-references are correct
- [ ] New content is linked from index
- [ ] Spelling and grammar are correct

## 📄 License

Documentation is licensed under the same terms as the AuraGen project.

---

For more information about AuraGen, visit the main [README](../README.md).
