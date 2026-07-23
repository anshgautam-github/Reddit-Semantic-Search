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


# =============================================================================
# 4.2.2 - Verify VSS Installation
# =============================================================================

def verify_vss_installation(extension_path: str = ".") -> bool:
    """
    Verify sqlite-vss extension loads and works correctly.

    Args:
        extension_path: Directory containing vss0 and vector0 extensions

    Returns:
        True if verification succeeds, False otherwise
    """
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)

    try:
        conn.load_extension(f"{extension_path}/vector0")
        conn.load_extension(f"{extension_path}/vss0")
        conn.enable_load_extension(False)

        result = conn.execute("""
            SELECT vss_version(),
                   vss_distance_l2(
                       vector_from_json('[1.0, 0.0, 0.0]'),
                       vector_from_json('[0.0, 1.0, 0.0]')
                   )
        """).fetchone()

        version, distance = result
        expected_distance = 1.414  # sqrt(2) for orthogonal unit vectors

        print(f"sqlite-vss version: {version}")
        print(f"L2 distance test: {distance:.3f} (expected ~{expected_distance:.3f})")

        conn.close()
        return abs(distance - expected_distance) < 0.01

    except Exception as e:
        print(f"Verification failed: {e}")
        conn.close()
        return False

