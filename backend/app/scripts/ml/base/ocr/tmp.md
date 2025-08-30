python predictWithOCR.py model='/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/best.pt' source='/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/demo.mp4'

(translate-env) (base) shhaofu@shhaofudeMacBook-Pro Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR % python predictWithOCR.py model='best.pt' source='demo.mp4'
Traceback (most recent call last):
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/predictWithOCR.py", line 7, in <module>
    from ultralytics.yolo.engine.predictor import BasePredictor
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/__init__.py", line 5, in <module>
    from ultralytics.hub import checks
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/hub/__init__.py", line 10, in <module>
    from ultralytics.hub.auth import Auth
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/hub/auth.py", line 5, in <module>
    from ultralytics.hub.utils import HUB_API_ROOT, request_with_credentials
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/hub/utils.py", line 10, in <module>
    from ultralytics.yolo.utils import DEFAULT_CONFIG_DICT, LOGGER, RANK, SETTINGS, TryExcept, colorstr, emojis
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/yolo/utils/__init__.py", line 17, in <module>
    import pandas as pd
  File "/opt/anaconda3/envs/translate-env/lib/python3.11/site-packages/pandas/__init__.py", line 22, in <module>
    from pandas.compat import is_numpy_dev as _is_numpy_dev  # pyright: ignore # noqa:F401
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/translate-env/lib/python3.11/site-packages/pandas/compat/__init__.py", line 18, in <module>
    from pandas.compat.numpy import (
  File "/opt/anaconda3/envs/translate-env/lib/python3.11/site-packages/pandas/compat/numpy/__init__.py", line 4, in <module>
    from pandas.util.version import Version
  File "/opt/anaconda3/envs/translate-env/lib/python3.11/site-packages/pandas/util/__init__.py", line 2, in <module>
    from pandas.util._decorators import (  # noqa:F401
  File "/opt/anaconda3/envs/translate-env/lib/python3.11/site-packages/pandas/util/_decorators.py", line 14, in <module>
    from pandas._libs.properties import cache_readonly
  File "/opt/anaconda3/envs/translate-env/lib/python3.11/site-packages/pandas/_libs/__init__.py", line 13, in <module>
    from pandas._libs.interval import Interval
  File "pandas/_libs/interval.pyx", line 1, in init pandas._libs.interval
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject


(pix2text) (base) shhaofu@shhaofudeMacBook-Pro Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR % python predictWithOCR.py model='best.pt' source='demo.mp4'
WARNING ⚠️ Different global settings detected, resetting to defaults. This may be due to an ultralytics package update. View and update your global settings directly in /Users/shhaofu/Library/Application Support/Ultralytics/settings.yaml
Downloading recognition model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% CompleteUltralytics YOLOv8.0.3 🚀 Python-3.9.0 torch-2.6.0 CPU
Error executing job with overrides: ['model=best.pt', 'source=demo.mp4']
Traceback (most recent call last):
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/predictWithOCR.py", line 112, in predict
    predictor()
  File "/opt/anaconda3/envs/pix2text/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/yolo/engine/predictor.py", line 164, in __call__
    model = self.model if self.done_setup else self.setup(source, model)
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/yolo/engine/predictor.py", line 121, in setup
    model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/nn/autobackend.py", line 73, in __init__
    model = attempt_load_weights(weights if isinstance(weights, list) else w,
  File "/Users/shhaofu/Code/Codes/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/nn/tasks.py", line 303, in attempt_load_weights
    ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
  File "/opt/anaconda3/envs/pix2text/lib/python3.9/site-packages/torch/serialization.py", line 1470, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default. Please use `torch.serialization.add_safe_globals([DetectionModel])` or the `torch.serialization.safe_globals([DetectionModel])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

(base) shhaofu@shhaofudeMacBook-Pro video % curl -u elastic:your_password_here -X PUT "https://localhost:9200/user_history" -H "Content-Type: application/json" -d'
          {
            "mappings": {
              "properties": {
                "user_id": { "type": "keyword" },
                "product_id": { "type": "keyword" },
                "view_time": { "type": "date" },
                "product_category": { "type": "keyword" },
                "product_tags": { "type": "keyword" }
              }
            }
          }' --insecure
{"acknowledged":true,"shards_acknowledged":false,"index":"user_history"}

{"errors":true,"took":60193,"items":[{"index":{"_index":"products","_id":"1","status":503,"error":{"type":"unavailable_shards_exception","reason":"[products][0] primary shard is not active Timeout: [1m], request: [BulkShardRequest [[products][0]] containing [5] requests]"}}},{"index":{"_index":"products","_id":"2","status":503,"error":{"type":"unavailable_shards_exception","reason":"[products][0] primary shard is not active Timeout: [1m], request: [BulkShardRequest [[products][0]] containing [5] requests]"}}},{"index":{"_index":"products","_id":"3","status":503,"error":{"type":"unavailable_shards_exception","reason":"[products][0] primary shard is not active Timeout: [1m], request: [BulkShardRequest [[products][0]] containing [5] requests]"}}},{"index":{"_index":"products","_id":"4","status":503,"error":{"type":"unavailable_shards_exception","reason":"[products][0] primary shard is not active Timeout: [1m], request: [BulkShardRequest [[products][0]] containing [5] requests]"}}},{"index":{"_index":"products","_id":"5","status":503,"error":{"type":"unavailable_shards_exception","reason":"[products][0] primary shard is not active Timeout: [1m], request: [BulkShardRequest [[products][0]] containing [5] requests]"}}}]}

agent发起的请求：
 1005  curl --insecure -u elastic:your_password_here -X GET https://localhost:9200/_cluster/health
 1006  curl --insecure -X GET http://localhost:9200
 1007  docker ps -a
 1008  docker logs es01
 1009  docker exec es01 cat /usr/share/elasticsearch/config/certs/http_ca.pem
 1010  docker exec es01 bash -c cat /usr/share/elasticsearch/config/elasticsearch.yml
 1011  docker exec es01 bash -c grep -A 10 'PASSWORD' /var/log/elasticsearch/docker-cluster.log

 的确是很有智能的感觉

 Created elasticsearch keystore in /usr/share/elasticsearch/config/elasticsearch.keystore

 sysbench oltp_read_write \
--db-driver=mysql \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=123123csd \
--mysql-db=pperformance \
--tables=10 \
--table-size=10000 \
prepare

sysbench oltp_read_write \
--db-driver=mysql \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=123123csd \
--mysql-db=pperformance \
--tables=10 \
--table-size=10000 \
--threads=4 \
--time=60 \
--report-interval=10 \
run


(base) shaofu@shaofu:~$ sudo systemctl status cron
[sudo] shaofu 的密码：
● cron.service - Regular background program processing daemon
     Loaded: loaded (/lib/systemd/system/cron.service; enabled; vendor preset: enabled)
     Active: active (running) since Wed 2025-03-12 08:59:59 CST; 1 month 0 days ago
       Docs: man:cron(8)
   Main PID: 760 (cron)
      Tasks: 1 (limit: 9289)
     Memory: 2.0M
        CPU: 1min 14.650s
     CGroup: /system.slice/cron.service
             └─760 /usr/sbin/cron -f -P

4月 12 10:27:01 shaofu cron[760]: (shaofu) RELOAD (crontabs/shaofu)
4月 12 10:27:01 shaofu CRON[151258]: pam_unix(cron:session): session opened for user root(uid=0) by (uid=0)
4月 12 10:27:01 shaofu CRON[151259]: pam_unix(cron:session): session opened for user shaofu(uid=1000) by (uid=0)
4月 12 10:27:01 shaofu CRON[151260]: (root) CMD (/home/shaofu/collect_processlist.sh)
4月 12 10:27:01 shaofu CRON[151261]: (shaofu) CMD (sudo /home/shoafu/collect_processlist.sh)
4月 12 10:27:01 shaofu sudo[151264]: pam_unix(sudo:auth): conversation failed
4月 12 10:27:01 shaofu sudo[151264]: pam_unix(sudo:auth): auth could not identify password for [shaofu]
4月 12 10:27:01 shaofu sudo[151264]:   shaofu : command not allowed ; PWD=/home/shaofu ; USER=root ; COMMAND=/home/shoafu/collect_processlist.sh
4月 12 10:27:01 shaofu CRON[151258]: pam_unix(cron:session): session closed for user root
4月 12 10:27:01 shaofu CRON[151259]: pam_unix(cron:session): session closed for user shaofu
4月 12 10:28:01 shaofu CRON[151520]: pam_unix(cron:session): session opened for user root(uid=0) by (uid=0)
4月 12 10:28:01 shaofu CRON[151521]: pam_unix(cron:session): session opened for user shaofu(uid=1000) by (uid=0)
4月 12 10:28:01 shaofu CRON[151522]: (root) CMD (/home/shaofu/collect_processlist.sh)
4月 12 10:28:01 shaofu CRON[151523]: (shaofu) CMD (sudo /home/shoafu/collect_processlist.sh)
4月 12 10:28:01 shaofu sudo[151526]: pam_unix(sudo:auth): conversation failed
4月 12 10:28:01 shaofu sudo[151526]: pam_unix(sudo:auth): auth could not identify password for [shaofu]
4月 12 10:28:01 shaofu sudo[151526]:   shaofu : command not allowed ; PWD=/home/shaofu ; USER=root ; COMMAND=/home/shoafu/collect_processlist.sh

