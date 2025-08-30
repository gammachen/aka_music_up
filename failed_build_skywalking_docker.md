(base) shhaofu@shhaofudeMacBook-Pro oap % docker build -t skywalking-oap .
[+] Building 4.5s (8/11)                                                                                                                                                          docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                                              0.0s
 => => transferring dockerfile: 1.42kB                                                                                                                                                            0.0s
 => [internal] load metadata for docker.io/library/eclipse-temurin:11-jre                                                                                                                         4.1s
 => [internal] load .dockerignore                                                                                                                                                                 0.0s
 => => transferring context: 2B                                                                                                                                                                   0.0s
 => [1/7] FROM docker.io/library/eclipse-temurin:11-jre@sha256:425f561992aed47c6575d0b52db6d90a86658dc6f06bbbf36d355b72432e88c5                                                                   0.1s
 => => resolve docker.io/library/eclipse-temurin:11-jre@sha256:425f561992aed47c6575d0b52db6d90a86658dc6f06bbbf36d355b72432e88c5                                                                   0.0s
 => => sha256:04c32eeb6473fbe85bf2276e158e3d15f90ed31b2288a53f7bad4f44be3c5dcf 1.94kB / 1.94kB                                                                                                    0.0s
 => => sha256:a97ec9065e193a886089d30e045ae40bac6b3031cd588c211e9dc37f5ee0e516 5.72kB / 5.72kB                                                                                                    0.0s
 => => sha256:425f561992aed47c6575d0b52db6d90a86658dc6f06bbbf36d355b72432e88c5 7.52kB / 7.52kB                                                                                                    0.0s
 => [internal] load build context                                                                                                                                                                 0.0s
 => => transferring context: 4.33kB                                                                                                                                                               0.0s
 => [2/7] WORKDIR /skywalking                                                                                                                                                                     0.0s
 => [3/7] COPY  .                                                                                                                                                                                 0.0s
 => ERROR [4/7] RUN set -ex;     tar -xzf "$DIST" --strip 1;     rm -rf "$DIST";     rm -rf "config/log4j2.xml";     rm -rf "bin";     rm -rf "webapp";     rm -rf "agent";     mkdir "bin";      0.2s
------
 > [4/7] RUN set -ex;     tar -xzf "$DIST" --strip 1;     rm -rf "$DIST";     rm -rf "config/log4j2.xml";     rm -rf "bin";     rm -rf "webapp";     rm -rf "agent";     mkdir "bin";:
0.159 + tar -xzf  --strip 1
0.163 tar (child): : Cannot open: No such file or directory
0.163 tar (child): Error is not recoverable: exiting now
0.164 tar: Child returned status 2
0.164 tar: Error is not recoverable: exiting now
------
Dockerfile:33
--------------------
  32 |
  33 | >>> RUN set -ex; \
  34 | >>>     tar -xzf "$DIST" --strip 1; \
  35 | >>>     rm -rf "$DIST"; \
  36 | >>>     rm -rf "config/log4j2.xml"; \
  37 | >>>     rm -rf "bin"; \
  38 | >>>     rm -rf "webapp"; \
  39 | >>>     rm -rf "agent"; \
  40 | >>>     mkdir "bin";
  41 |
--------------------
ERROR: failed to solve: process "/bin/sh -c set -ex;     tar -xzf \"$DIST\" --strip 1;     rm -rf \"$DIST\";     rm -rf \"config/log4j2.xml\";     rm -rf \"bin\";     rm -rf \"webapp\";     rm -rf \"agent\";     mkdir \"bin\";" did not complete successfully: exit code: 2

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/v5gk9qdprs8bv9tzdnhk2ny4i


