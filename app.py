# Reddit personal knowledge base with SQLite-VSS vector search.

# Dependencies: pip install praw sentence-transformers numpy
# Also requires sqlite-vss binaries (vector0, vss0) - see README.md


import sqlite3
import json
import re
import time
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Union
from dataclasses import dataclass
from datetime import datetime

import praw
from prawcore.exceptions import PrawcoreException
from sentence_transformers import SentenceTransformer

