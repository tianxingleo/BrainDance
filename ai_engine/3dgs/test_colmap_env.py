import subprocess
import os

env = os.environ.copy()
cmd_env = env.copy()

for env_var in ["LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH"]:
    if env_var in cmd_env:
        del cmd_env[env_var]

cmd = ["/usr/local/bin/colmap", "feature_extractor", "--database_path", "/tmp/colmap_test/db.db", "--image_path", "/tmp/colmap_test/images", "--FeatureExtraction.use_gpu", "0"]
print("Running COLMAP with env mod:")
process = subprocess.Popen(cmd, env=cmd_env)
process.wait()
print("Return code:", process.returncode)
