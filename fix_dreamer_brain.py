"""
Fix the dreamer_brain bootstrap by extracting and patching relative imports.
"""
import sys
import base64
import gzip
import re

# Get the embedded source from champion_gen42
sys.path.insert(0, '.')

# Read champion_gen42 directly to extract _DREAMER_BRAIN_SOURCE
with open('champion_gen42.py', 'rb') as f:
    content = f.read()

# Find _DREAMER_BRAIN_SOURCE
import ast

# Since file is huge, let's grep for it
text = content.decode('utf-8', errors='ignore')

# Find the line with _DREAMER_BRAIN_SOURCE
match = re.search(r'_DREAMER_BRAIN_SOURCE\s*=\s*"([^"]+)"', text)
if match:
    b64_source = match.group(1)
    print(f"Found _DREAMER_BRAIN_SOURCE: {len(b64_source)} chars")
    
    try:
        compressed = base64.b64decode(b64_source)
        source = gzip.decompress(compressed).decode('utf-8')
        print(f"\nDecompressed source: {len(source)} chars")
        print("\n=== FIRST 2000 CHARS ===")
        print(source[:2000])
        print("\n=== LOOKING FOR RELATIVE IMPORTS ===")
        for line in source.split('\n'):
            if 'from .' in line or 'import .' in line:
                print(f"  RELATIVE: {line}")
    except Exception as e:
        print(f"Decompress failed: {e}")
else:
    print("Could not find _DREAMER_BRAIN_SOURCE")
    
    # Try finding the variable another way
    match2 = re.search(r'_DREAMER_BRAIN_SOURCE\s*=', text)
    if match2:
        print(f"Found at position {match2.start()}")
        print(text[match2.start():match2.start()+500])
