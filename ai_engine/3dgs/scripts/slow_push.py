import subprocess
import time
import sys
from datetime import datetime

REMOTE_NAME = "origin"
CHECK_INTERVAL = 60
PUSH_INTERVAL = 3600
TARGET_BRANCH = "tianxingleo"
PUSH_TIMEOUT = 300
MIN_INTERVAL = 3600

def run_cmd(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

def get_current_branch():
    return run_cmd("git rev-parse --abbrev-ref HEAD")

def wait_for_target_branch():
    while True:
        branch = get_current_branch()
        if branch == TARGET_BRANCH:
            return branch
        print(f"[{datetime.now()}] 当前在分支 '{branch}'，等待切换到 '{TARGET_BRANCH}'...")
        time.sleep(CHECK_INTERVAL)

def get_unpushed_commits(branch):
    run_cmd(f"git fetch {REMOTE_NAME} {branch}")
    cmd = f"git rev-list --reverse {REMOTE_NAME}/{branch}..HEAD"
    output = run_cmd(cmd)
    if not output:
        return []
    return output.splitlines()

def get_commit_timestamp(commit_hash):
    cmd = f"git log -1 --format=%ct {commit_hash}"
    output = run_cmd(cmd)
    if output:
        return int(output)
    return None

def wait_for_min_interval(commit_hash):
    timestamp = get_commit_timestamp(commit_hash)
    if timestamp is None:
        return
    elapsed = time.time() - timestamp
    wait_time = MIN_INTERVAL - elapsed
    if wait_time > 0:
        print(f"⏰ 上一个Commit距今 {int(elapsed)} 秒，需等待 {int(wait_time)} 秒才能推送...")
        time.sleep(wait_time)
    else:
        print(f"✅ 上一个Commit距今 {int(elapsed)} 秒，已超过1小时限制")

def main():
    print(f"[{datetime.now()}] 🚀 Slow Push 脚本启动...")
    print(f"目标分支: {TARGET_BRANCH}")
    print(f"推送间隔: {PUSH_INTERVAL/3600} 小时")

    while True:
        try:
            branch = wait_for_target_branch()
            print(f"[{datetime.now()}] ✅ 已切换到分支 '{branch}'")

            commits = get_unpushed_commits(branch)

            if not commits:
                print(f"[{datetime.now()}] 😴 无待推送 Commit，{CHECK_INTERVAL}秒后重试...", end="\r")
                time.sleep(CHECK_INTERVAL)
                continue

            next_commit = commits[0]
            remaining = len(commits) - 1

            print(f"\n[{datetime.now()}] 💡 发现 {len(commits)} 个待推送 Commit")
            print(f"👉 推送 Commit: {next_commit[:7]} -> {REMOTE_NAME}/{branch}")

            wait_for_min_interval(next_commit)

            push_cmd = f"git push {REMOTE_NAME} {next_commit}:{branch}"
            print(f"⏳ 开始推送 (超时: {PUSH_TIMEOUT}秒)...")
            ret = subprocess.call(push_cmd, shell=True, timeout=PUSH_TIMEOUT)

            if ret == 0:
                print(f"✅ 推送成功！剩余: {remaining} 个")
                print(f"⏳ 等待 {PUSH_INTERVAL} 秒...")
                time.sleep(PUSH_INTERVAL)
            else:
                print("❌ 推送失败，5分钟后重试...")
                time.sleep(300)

        except subprocess.TimeoutExpired:
            print(f"⏰ 推送超时 ({PUSH_TIMEOUT}秒)，5分钟后重试...")
            time.sleep(300)
        except KeyboardInterrupt:
            print("\n🛑 脚本已停止")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
