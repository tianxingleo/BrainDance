import os
import subprocess
import shutil

cuda_lib_path = "/usr/local/cuda/lib64" if os.path.exists("/usr/local/cuda/lib64") else ""
ld_library_paths = [cuda_lib_path, "/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib", "/lib"]
ld_library_paths = [p for p in ld_library_paths if p] # filter empty
clean_env = {
    "PATH": os.pathsep.join(["/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin"]),
    "LD_LIBRARY_PATH": os.pathsep.join(ld_library_paths),
    "HOME": os.getenv("HOME", ""),
    "USER": os.getenv("USER", ""),
    "LANG": os.getenv("LANG", "en_US.UTF-8"),
    "SHELL": os.getenv("SHELL", "/bin/bash"),
    "TERM": os.getenv("TERM", "xterm-256color")
}
cmd = ["/usr/local/bin/glomap", "mapper", "--database_path", "/tmp/glomap2/db.db", "--image_path", "/tmp/glomap2", "--output_path", "/tmp/glomap2/out", "--GlobalPositioning.use_gpu", "1"]
process = subprocess.Popen(cmd, env=clean_env)
process.wait()
print("Return code:", process.returncode)
