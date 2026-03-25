import subprocess
import os

clean_env = {
    "PATH": os.pathsep.join(["/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin"]),
    "LD_LIBRARY_PATH": os.pathsep.join(["/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib", "/lib"]),
    "HOME": os.getenv("HOME", ""),
    "USER": os.getenv("USER", ""),
    "LANG": os.getenv("LANG", "en_US.UTF-8"),
    "SHELL": os.getenv("SHELL", "/bin/bash"),
    "TERM": os.getenv("TERM", "xterm-256color")
}
cmd = ["/usr/local/bin/colmap", "feature_extractor", "--database_path", "/tmp/colmap_test/db.db", "--image_path", "/tmp/colmap_test/images", "--FeatureExtraction.use_gpu", "1"]
print("Running with GPU=1:")
process = subprocess.Popen(cmd, env=clean_env)
process.wait()
print("Return code:", process.returncode)
