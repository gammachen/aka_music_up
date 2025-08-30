import random
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

# 连接 InfluxDB
client = InfluxDBClient(
    url="http://localhost:8086",
    token="iXKclTXCS-sTHoP2QgUt--UtVbJHog9bs5zjYgD16uNozmD36cDvuwUH_tWQ_F9CAfhh6QmiryV31hSPtuD_3g==",  # 用户名和密码生成的 token（这里简化处理）
    org="csd"  # 默认 org
)
bucket = "csd"

# 生成模拟数据
def generate_data():
    exhibitions = [
        {"exhibition_id": "exhibition_001", "location": "gallery_A"},
        {"exhibition_id": "exhibition_002", "location": "gallery_B"},
        {"exhibition_id": "exhibition_003", "location": "gallery_C"},
    ]
    
    while True:
        for ex in exhibitions:
            current_people = random.randint(10, 100)
            temperature = round(random.uniform(20.0, 30.0), 1)
            
            point = Point("exhibition_traffic") \
                .tag("exhibition_id", ex["exhibition_id"]) \
                .tag("location", ex["location"]) \
                .field("current_people", current_people) \
                .field("temperature", temperature) \
                .time(time.time_ns())
            
            write_api = client.write_api(write_options=SYNCHRONOUS)
            write_api.write(bucket, record=point)
            print(f"Sent data for {ex['exhibition_id']}")
        time.sleep(5)  # 每5秒生成一次数据

if __name__ == "__main__":
    generate_data()