```shell
(base) shhaofu@shhaofudeMacBook-Pro nginx-rtmp-monitoring % chmod +x setup.sh && ./setup.sh
WARN[0000] /Users/shhaofu/Code/Codes/nginx-rtmp-monitoring/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion
[+] Running 10/10
 ✔ nginx_rtmp Pulled                                                                                                                                                                              38.7s
   ✔ 530afca65e2e Pull complete                                                                                                                                                                    3.5s
   ✔ c20ba4fc8a2c Pull complete                                                                                                                                                                    8.2s
   ✔ 7d60e766d5fd Pull complete                                                                                                                                                                    8.5s
   ✔ 13fcd85631ea Pull complete                                                                                                                                                                    8.5s
   ✔ 58b7d31a1710 Pull complete                                                                                                                                                                   13.1s
   ✔ f1d974888e5e Pull complete                                                                                                                                                                   13.1s
   ✔ 3cf0b843d524 Pull complete                                                                                                                                                                   13.1s
   ✔ c99ac4327a57 Pull complete                                                                                                                                                                   13.2s
   ✔ 6a0e61809b5b Pull complete                                                                                                                                                                   13.2s
[+] Building 36.5s (2/2) FINISHED                                                                                                                                                  docker:desktop-linux
 => [dashboard internal] load build definition from Dockerfile                                                                                                                                     0.0s
 => => transferring dockerfile: 254B                                                                                                                                                               0.0s
 => ERROR [dashboard internal] load metadata for docker.io/library/node:argon-slim                                                                                                                36.4s
------
 > [dashboard internal] load metadata for docker.io/library/node:argon-slim:
------
failed to solve: node:argon-slim: failed to resolve source metadata for docker.io/library/node:argon-slim: failed to do request: Head "https://docker.mirrors.ustc.edu.cn/v2/library/node/manifests/argon-slim?ns=docker.io": EOF
```

