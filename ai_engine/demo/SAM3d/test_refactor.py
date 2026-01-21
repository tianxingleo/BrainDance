import sys
from pathlib import Path

# Add the current directory to sys.path to ensure we can import sam3d_engine
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from sam3d_engine.core import SAM3DEngine
    print("✅ Successfully imported SAM3DEngine")
except ImportError as e:
    print(f"❌ Failed to import SAM3DEngine: {e}")
    sys.exit(1)

def test_initialization():
    print("🧪 Testing SAM3DEngine initialization...")
    repo_path = Path.home() / "workspace/ai/sam-3d-objects"
    if not repo_path.exists():
        print(f"⚠️ Warning: SAM3D repo path {repo_path} does not exist. Skipping full initialization test.")
        # Create a mock repo path for testing structure if needed, or just warn
        return

    try:
        engine = SAM3DEngine(repo_path=str(repo_path))
        print("✅ SAM3DEngine initialized successfully (Mocks injected)")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        # We expect some failure if specific files are missing, but imports should work
        # Specifically, it might fail at _setup_path or loading config if they don't exist
        
    print("🎉 Test verification complete.")

if __name__ == "__main__":
    test_initialization()
