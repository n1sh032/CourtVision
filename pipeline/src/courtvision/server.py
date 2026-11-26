"""Serve the review UI and run one local detection job at a time."""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
