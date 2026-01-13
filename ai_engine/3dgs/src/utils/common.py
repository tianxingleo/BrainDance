# [工具函数] 存放 format_duration
import datetime

def format_duration(seconds):

    """
    [辅助函数] 将秒数转换为易读的 HH:MM:SS 格式
    """
    # [标准库] datetime.timedelta 自动处理时间换算（如 3661秒 -> 1:01:01）
    return str(datetime.timedelta(seconds=int(seconds)))