
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

try:
    print("Attempting to import app...")
    from app import build_graph
    print("Import successful. Attempting to build graph...")
    graph = build_graph()
    print("Graph built successfully!")
except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
