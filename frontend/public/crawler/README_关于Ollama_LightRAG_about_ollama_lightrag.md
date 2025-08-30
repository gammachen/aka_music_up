```shell
on ollama console:
set parameter num_ctx 32768

ollama show --modelfile qwen2 > Modelfile

(lightrag) shhaofu@shhaofudeMacBook-Pro LightRAG % ollama list
NAME                       ID              SIZE      MODIFIED
nomic-embed-text:latest    0a109f422b47    274 MB    22 seconds ago
llama3:latest              365c0bd3c000    4.7 GB    6 months ago
llava:latest               8dd30f6b0cb1    4.7 GB    6 months ago
qwen2:latest               e0d4e1163c58    4.4 GB    7 months ago
qwen:14b                   80362ced6553    8.2 GB    7 months ago
orca-mini:latest           2dbd9f439647    2.0 GB    11 months ago
qwen:7b                    2091ee8c8d8f    4.5 GB    11 months ago
llama2:latest              78e26419b446    3.8 GB    12 months ago
(lightrag) shhaofu@shhaofudeMacBook-Pro LightRAG % ollama run qwen2
>>> /set
Available Commands:
  /set parameter ...     Set a parameter
  /set system <string>   Set system message
  /set history           Enable history
  /set nohistory         Disable history
  /set wordwrap          Enable wordwrap
  /set nowordwrap        Disable wordwrap
  /set format json       Enable JSON mode
  /set noformat          Disable formatting
  /set verbose           Show LLM stats
  /set quiet             Disable LLM stats

>>> /set prameter num_ctx 32000
Unknown command '/set prameter'. Type /? for help
>>> /set parameter num_ctx 32000
Set parameter 'num_ctx' to '32000'
>>> /bye
(lightrag) shhaofu@shhaofudeMacBook-Pro LightRAG % ollama show --modelfile qwen2 > Modelfile
(lightrag) shhaofu@shhaofudeMacBook-Pro LightRAG % vi Modelfile
(lightrag) shhaofu@shhaofudeMacBook-Pro LightRAG % ollama create -f Modelfile qwen2m
gathering model components
copying file sha256:43f7a214e5329f672bb05404cfba1913cbb70fdaa1a17497224e1925046b0ed5 100%
parsing GGUF
using existing layer sha256:43f7a214e5329f672bb05404cfba1913cbb70fdaa1a17497224e1925046b0ed5
using existing layer sha256:62fbfd9ed093d6e5ac83190c86eec5369317919f4b149598d2dbb38900e9faef
using existing layer sha256:c156170b718ec29139d3653d40ed1986fd92fb7e0959b5c71f3c48f62e6636f4
creating new layer sha256:90976b326ae6da3a73f53a9eaa645c123a99aed6f62b12a06c8c4b121555700d
writing manifest
success

(base) shhaofu@shhaofudeMacBook-Pro ~ % ollama ps
NAME             ID              SIZE      PROCESSOR    UNTIL
qwen2m:latest    09450a55d3f9    8.7 GB    100% GPU     About a minute from now
```


```shell
curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "who is gammachen", "mode": "local"}'

/Users/shhaofu/Code/aigc-ml-light-0/static/nsfw/ham_bk/yi_di_ji_mao.txt

```

```shell

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "who is Mr. Scrooge", "mode": "hybrid"}'
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "who is Mr. Scrooge, what is the relationship to the other characters?", "mode": "local"}'
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "who is gammachen", "mode": "local"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "钰慧不在的第一个周末，我早上还有一些事情处理，打算傍晚过後再搭飞机去台南。中午的时候我办完事刚回到家，隔壁的姚太太跑来找我"}'

curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "谢太太今天穿着很正式的上班套装，短外套和短裙都是鹅黄色的，白色的丝质圆荷叶领衬衫，自然的贴在丰满的乳房上，我相信她那内衣也是白色的。短裙下露出雪白的大腿，隔着丝袜，可以看得见腿的皮肤应该是非常光滑细致的"}'

curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "我知道她等会儿还要参加公司的活动，衣服乱了不好，便不再摸她的胸，但是我倒又摸起她的腿来了。我沿着大腿内侧往上摸，发现她的腿在不停的颤抖，我终於摸到了那满涨的顶端，用手指轻轻的按动，那个敏感的地方传来她温暖的体温，而且有一点点湿润"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/file.txt"
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@./book.txt"

curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@/Users/shhaofu/Code/aigc-ml-light-0/static/nsfw/ham_bk/yi_di_ji_mao.txt"

curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@./book.txt"

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"

curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "who is Mr. Scrooge, what is the relationship to the other characters?", "mode": "local"}'

curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "who is gammachen", "mode": "local"}'


curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "my name is gammachen, i am an prompt enginner working in SPACE, welcome to my home"}'
```



