MODULE: Interface.panels.image_viewer

PURPOSE:
Image Viewer panel — display images in the terminal (fallback to metadata if no renderer).

PUBLIC_API:


DEPENDENCIES:
- PIL
- __future__
- os
- pathlib
- rich.panel
- rich.style
- rich.text
- textual.app
- textual.binding
- textual.containers
- textual.widgets
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
