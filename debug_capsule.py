"""Debug capsule bootstrap"""
import sys
import base64
import gzip
import types
import traceback

# Clear any existing modules that might conflict
for mod in list(sys.modules.keys()):
    if mod in ['config', 'brain', 'dreamer_brain'] or mod.startswith('dreamerv3') or mod.startswith('embodied'):
        del sys.modules[mod]

print("=== Debugging Capsule Bootstrap ===")
print()

# Load the capsule
capsule_path = 'F:/End-Game/glassboxgames/children/key-data-repo/models/champion_gen52.py'

# Read the file
with open(capsule_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"Capsule size: {len(content)} bytes")
print()

# Find and extract _DREAMERV3_DEPS
# Look for the dict definition
start_marker = "_DREAMERV3_DEPS = {"
start_idx = content.find(start_marker)
if start_idx == -1:
    print("ERROR: _DREAMERV3_DEPS not found")
    sys.exit(1)

# Find matching closing brace
brace_count = 0
end_idx = start_idx
for i in range(start_idx, len(content)):
    if content[i] == '{':
        brace_count += 1
    elif content[i] == '}':
        brace_count -= 1
        if brace_count == 0:
            end_idx = i + 1
            break

deps_code = content[start_idx:end_idx]
print(f"Found _DREAMERV3_DEPS, {len(deps_code)} chars")

# Execute to get the dict
local_ns = {}
exec(deps_code, local_ns)
_DREAMERV3_DEPS = local_ns['_DREAMERV3_DEPS']

print(f"Keys: {list(_DREAMERV3_DEPS.keys())}")
print()

# Try to decompress each embedded module
for module_path, b64_source in _DREAMERV3_DEPS.items():
    print(f"--- {module_path} ---")
    try:
        compressed = base64.b64decode(b64_source)
        source = gzip.decompress(compressed).decode('utf-8')
        print(f"  Decompressed OK: {len(source)} bytes")
        
        # Check for 'get' function in config.py
        if module_path == 'config.py':
            if 'def get(' in source:
                print("  ✓ Contains 'def get()' function")
            else:
                print("  ✗ Missing 'def get()' function!")
            print(f"  First 300 chars: {source[:300]}")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
    print()

# Now try to actually bootstrap
print("=== Attempting Bootstrap ===")

# CRITICAL: config MUST be bootstrapped BEFORE brain (brain imports from config)
priority_modules = ['config', 'brain']  # Fixed order!
for module_name in priority_modules:
    module_path = module_name + '.py'
    if module_path in _DREAMERV3_DEPS and module_name not in sys.modules:
        print(f"Bootstrapping {module_name}...")
        try:
            b64_source = _DREAMERV3_DEPS[module_path]
            compressed = base64.b64decode(b64_source)
            source = gzip.decompress(compressed).decode('utf-8')
            
            module = types.ModuleType(module_name)
            module.__file__ = '<embedded:' + module_path + '>'
            exec(compile(source, '<' + module_name + '>', 'exec'), module.__dict__)
            sys.modules[module_name] = module
            
            print(f"  ✓ {module_name} bootstrapped")
            if module_name == 'config':
                print(f"    config.get = {getattr(module, 'get', 'NOT FOUND')}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()
    else:
        print(f"Skipping {module_name} (already in sys.modules or not in deps)")

print()
print("=== Checking sys.modules ===")
print(f"config in sys.modules: {'config' in sys.modules}")
print(f"brain in sys.modules: {'brain' in sys.modules}")

if 'config' in sys.modules:
    config = sys.modules['config']
    print(f"config.get: {getattr(config, 'get', 'NOT FOUND')}")
    print(f"config.__file__: {getattr(config, '__file__', 'NOT SET')}")
