# Vendored browser libraries (optional)

The viewer loads Cytoscape.js from this directory first, then from the public CDNs
(cdnjs, jsDelivr). For a fully offline installation, place the library here:

    biomedcat/webapp/static/vendor/cytoscape.min.js

taken from https://github.com/cytoscape/cytoscape.js/releases (dist/cytoscape.min.js, MIT license,
version 3.30 or later). Nothing else is required: the console itself has no other browser dependency.
