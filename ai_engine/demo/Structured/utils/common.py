import subprocess
import logging
import sys
from config.settings import ENV

def setup_logging():
    # 屏蔽 Nerfstudio 的冗余日志
    logging.getLogger('nerfstudio').setLevel(logging.ERROR)

def run_command(cmd, check=True, capture_output=False, env=ENV):
    """统一执行 shell 命令的封装"""
    try:
        # 如果是 capture_output，我们需要处理 text=True
        text_mode = True if capture_output else False
        
        result = subprocess.run(
            cmd,
            check=check,
            env=env,
            capture_output=capture_output,
            text=text_mode
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 命令执行失败: {' '.join([str(x) for x in cmd])}")
        if capture_output:
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
        raise e
