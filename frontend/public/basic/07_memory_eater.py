import sys
import time

def memory_hog():
    """
    持续分配内存，直到物理内存耗尽，触发Swap空间使用。
    """
    data = []  # 存储大对象的列表
    chunk_size = 100 * 1024 * 1024  # 每次分配1MB（字节）

    while True:
        try:
            # 创建一个1MB的字节对象
            byte_array = bytearray(chunk_size)
            data.append(byte_array)

            # 打印当前内存使用情况
            # current_usage = sys.getsizeof(data) // (1024 * 1024)
            current_usage = len(data) * chunk_size // (1024 * 1024)
            print(f"Allocated {current_usage} MB, Total: {len(data)} chunks")

            # 每秒检查一次内存
            time.sleep(1)
        except MemoryError:
            print("MemoryError: Out of memory! Continuing...")
            time.sleep(1)
            continue

if __name__ == "__main__":
    print("Starting memory hog process...")
    print("WARNING: This will consume all available memory and Swap space.")
    print("Run this only in a controlled environment (e.g., VM).")
    memory_hog()