#!/usr/bin/env python
"""Punto de entrada de la herramienta de detección rápida de neumonía.

Delega toda la interfaz y la lógica en los módulos de ``src`` para mantener
una única fuente de responsabilidad por módulo.
"""

from __future__ import annotations

from src.ui import main

if __name__ == "__main__":
    raise SystemExit(main())
