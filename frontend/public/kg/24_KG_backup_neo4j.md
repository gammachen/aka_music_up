## docker中进行备份

```shell
neo4j-admin database dump --verbose --to-path=/data/backup neo4j

# neo4j-admin database dump --verbose --to-path=/data/backup neo4j
Executing command line: /opt/java/openjdk/bin/java -cp /var/lib/neo4j/plugins/*:/var/lib/neo4j/conf/*:/var/lib/neo4j/lib/* -Dfile.encoding=UTF-8 org.neo4j.cli.AdminTool database dump --verbose --to-path=/data/backup neo4j
neo4j 5.25.1
VM Name: OpenJDK 64-Bit Server VM
VM Vendor: Eclipse Adoptium
VM Version: 17.0.13+11
JIT compiler: HotSpot 64-Bit Tiered Compilers
VM Arguments: [-Dfile.encoding=UTF-8]
Configuration files used (ordered by priority):
/var/lib/neo4j/conf/neo4j.conf
--------------------
2024-11-06 01:10:53.902+0000 INFO  [o.n.c.d.DumpCommand] Starting dump of database 'neo4j'
Done: 39 files, 724.8MiB processed in 14.930 seconds.
2024-11-06 01:11:10.052+0000 INFO  [o.n.c.d.DumpCommand] Dump completed successfully
# ls
backup  databases  server_id  transactions
# cd backup
# ls
neo4j.dump
# ls -alh
total 209M
drwxr-xr-x 3 root  root    96 Nov  6 01:10 .
drwxr-xr-x 7 neo4j neo4j  224 Nov  6 01:10 ..
-rw-r--r-- 1 root  root  197M Nov  6 01:11 neo4j.dump
```

```shell
拷贝到本地：
docker cp c5048d796f83049d1f7d14f1b5b8ff9be5ed5fb8466f47d31c080e17cfd3cc16:/data/backup .

shaofu@shaofu-Aspire-4741:~/backup_of_neo4j$ sudo docker cp neo4j.dump 0df62351e774843bd9631b985709075954603128f33d05829f245a26b915c643:/data
Successfully copied 207MB to 0df62351e774843bd9631b985709075954603128f33d05829f245a26b915c643:/data
shaofu@shaofu-Aspire-4741:~/backup_of_neo4j$ sudo docker exec -it 0df62351e774843bd9631b985709075954603128f33d05829f245a26b915c643 /bin/bash
```

```shell
(base) shaofu@shaofu:~$ neo4j-admin database dump --verbose --to-path=/home/shaofu/backup_neo4j_samples neo4j
Executing command line: /usr/lib/jvm/java-21-openjdk-amd64/bin/java -cp /var/lib/neo4j/plugins/*:/etc/neo4j/*:/usr/share/neo4j/lib/* -XX:+UseParallelGC -XX:-OmitStackTraceInFastThrow -XX:+UnlockExperimentalVMOptions -XX:+TrustFinalNonStaticFields -XX:+DisableExplicitGC -Djdk.nio.maxCachedBufferSize=1024 -Dio.netty.tryReflectionSetAccessible=true -XX:+ExitOnOutOfMemoryError -Djdk.tls.ephemeralDHKeySize=2048 -XX:FlightRecorderOptions=stackdepth=256 -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED -Dlog4j2.disable.jmx=true -Xlog:gc*,safepoint,age*=trace:file=/var/log/neo4j/gc.log::filecount=5,filesize=20480k -Dfile.encoding=UTF-8 org.neo4j.cli.AdminTool database dump --verbose --to-path=/home/shaofu/backup_neo4j_samples neo4j
Could not rename log file '/var/log/neo4j/gc.log' to '/var/log/neo4j/gc.log.1' (Permission denied).
[0.001s][error][logging] Error opening log file '/var/log/neo4j/gc.log': Permission denied
[0.001s][error][logging] Initialization of output 'file=/var/log/neo4j/gc.log' using options 'filecount=5,filesize=20480k' failed.
Invalid -Xlog option '-Xlog:gc*,safepoint,age*=trace:file=/var/log/neo4j/gc.log::filecount=5,filesize=20480k', see error log for details.
Error: Could not create the Java Virtual Machine.
Error: A fatal exception has occurred. Program will exit.
(base) shaofu@shaofu:~$ sudo neo4j-admin database dump --verbose --to-path=/home/shaofu/backup_neo4j_samples neo4j
[sudo] shaofu 的密码：
Executing command line: /usr/lib/jvm/java-21-openjdk-amd64/bin/java -cp /var/lib/neo4j/plugins/*:/etc/neo4j/*:/usr/share/neo4j/lib/* -XX:+UseParallelGC -XX:-OmitStackTraceInFastThrow -XX:+UnlockExperimentalVMOptions -XX:+TrustFinalNonStaticFields -XX:+DisableExplicitGC -Djdk.nio.maxCachedBufferSize=1024 -Dio.netty.tryReflectionSetAccessible=true -XX:+ExitOnOutOfMemoryError -Djdk.tls.ephemeralDHKeySize=2048 -XX:FlightRecorderOptions=stackdepth=256 -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED -Dlog4j2.disable.jmx=true -Xlog:gc*,safepoint,age*=trace:file=/var/log/neo4j/gc.log::filecount=5,filesize=20480k -Dfile.encoding=UTF-8 org.neo4j.cli.AdminTool database dump --verbose --to-path=/home/shaofu/backup_neo4j_samples neo4j
neo4j 5.22.0
VM Name: OpenJDK 64-Bit Server VM
VM Vendor: Ubuntu
VM Version: 21.0.4+7-Ubuntu-1ubuntu222.04
JIT compiler: HotSpot 64-Bit Tiered Compilers
VM Arguments: [-XX:+UseParallelGC, -XX:-OmitStackTraceInFastThrow, -XX:+UnlockExperimentalVMOptions, -XX:+TrustFinalNonStaticFields, -XX:+DisableExplicitGC, -Djdk.nio.maxCachedBufferSize=1024, -Dio.netty.tryReflectionSetAccessible=true, -XX:+ExitOnOutOfMemoryError, -Djdk.tls.ephemeralDHKeySize=2048, -XX:FlightRecorderOptions=stackdepth=256, -XX:+UnlockDiagnosticVMOptions, -XX:+DebugNonSafepoints, --add-opens=java.base/java.nio=ALL-UNNAMED, --add-opens=java.base/java.io=ALL-UNNAMED, --add-opens=java.base/sun.nio.ch=ALL-UNNAMED, -Dlog4j2.disable.jmx=true, -Xlog:gc*,safepoint,age*=trace:file=/var/log/neo4j/gc.log::filecount=5,filesize=20480k, -Dfile.encoding=UTF-8]
Configuration files used (ordered by priority):
/etc/neo4j/neo4j-admin.conf
/etc/neo4j/neo4j.conf
--------------------
2024-11-06 01:22:46.855+0000 INFO  [o.n.c.d.DumpCommand] Starting dump of database 'neo4j'
2024-11-06 01:22:47.119+0000 ERROR [o.n.c.d.DumpCommand] Failed to dump database 'neo4j': The database is in use. Stop database 'neo4j' and try again.
2024-11-06 01:22:47.128+0000 ERROR [o.n.c.d.DumpCommand] Dump failed for databases: 'neo4j'
org.neo4j.cli.CommandFailedException: Dump failed for databases: 'neo4j'
	at org.neo4j.commandline.dbms.DumpCommand.execute(DumpCommand.java:191)
	at org.neo4j.cli.AbstractCommand.call(AbstractCommand.java:92)
	at org.neo4j.cli.AbstractCommand.call(AbstractCommand.java:37)
	at picocli.CommandLine.executeUserObject(CommandLine.java:2045)
	at picocli.CommandLine.access$1500(CommandLine.java:148)
	at picocli.CommandLine$RunLast.executeUserObjectOfLastSubcommandWithSameParent(CommandLine.java:2465)
	at picocli.CommandLine$RunLast.handle(CommandLine.java:2457)
	at picocli.CommandLine$RunLast.handle(CommandLine.java:2419)
	at picocli.CommandLine$AbstractParseResultHandler.execute(CommandLine.java:2277)
	at picocli.CommandLine$RunLast.execute(CommandLine.java:2421)
	at picocli.CommandLine.execute(CommandLine.java:2174)
	at org.neo4j.cli.AdminTool.execute(AdminTool.java:94)
	at org.neo4j.cli.AdminTool.main(AdminTool.java:82)
Caused by: org.neo4j.cli.CommandFailedException: The database is in use. Stop database 'neo4j' and try again.
	at org.neo4j.commandline.dbms.DumpCommand.execute(DumpCommand.java:170)
	... 12 more
Caused by: org.neo4j.io.locker.FileLockException: Lock file has been locked by another process: /var/lib/neo4j/data/databases/neo4j/database_lock. Please ensure no other process is using this database, and that the directory is writable (required even for read-only access)
	at org.neo4j.io.locker.Locker.storeLockException(Locker.java:151)
	at org.neo4j.io.locker.Locker.checkLock(Locker.java:82)
	at org.neo4j.kernel.internal.locker.GlobalFileLocker.checkLock(GlobalFileLocker.java:55)
	at org.neo4j.kernel.internal.locker.DatabaseLocker.checkLock(DatabaseLocker.java:28)
	at org.neo4j.commandline.dbms.LockChecker.checkLock(LockChecker.java:81)
	at org.neo4j.commandline.dbms.LockChecker.check(LockChecker.java:65)
	at org.neo4j.commandline.dbms.LockChecker.checkDatabaseLock(LockChecker.java:53)
	at org.neo4j.commandline.dbms.DumpCommand.execute(DumpCommand.java:166)
	... 12 more
```

```shell
(base) shaofu@shaofu:~$ sudo neo4j -h
Usage: neo4j [-hV] [--expand-commands] [--verbose] [COMMAND]
A partial alias for 'neo4j-admin server'. Commands for working with DBMS process from 'neo4j-admin server' category can
be invoked using this command.
      --expand-commands   Allow command expansion in config value evaluation.
  -h, --help              Show this help message and exit.
  -V, --version           Print version information and exit.
      --verbose           Prints additional information.
Commands:
  version  Print version information and exit.
  help     Display help information about the specified command.
  console  Start server in console.
  restart  Restart the server daemon.
  start    Start server as a daemon.
  status   Get the status of the neo4j server process.
  stop     Stop the server daemon.

Environment variables:
  NEO4J_CONF    Path to directory which contains neo4j.conf.
  NEO4J_DEBUG   Set to anything to enable debug output.
  NEO4J_HOME    Neo4j home directory.
  HEAP_SIZE     Set JVM maximum heap size during command execution. Takes a number and a unit, for example 512m.
  JAVA_OPTS     Used to pass custom setting to Java Virtual Machine executing the command. Refer to JVM documentation
about the exact format. This variable is incompatible with HEAP_SIZE and takes precedence over HEAP_SIZE.
```

```shell
Docker中启动的neo4j的dump操作无需停止服务，直接执行即可。但是在ubuntu下需要停止服务，等于说是离线备份

(base) shaofu@shaofu:~$ sudo neo4j status
Neo4j is running at pid 6052
(base) shaofu@shaofu:~$ sudo neo4j stop
Stopping Neo4j......... stopped.
(base) shaofu@shaofu:~$ sudo neo4j-admin database dump --verbose --to-path=/home/shaofu/backup_neo4j_samples neo4j
Executing command line: /usr/lib/jvm/java-21-openjdk-amd64/bin/java -cp /var/lib/neo4j/plugins/*:/etc/neo4j/*:/usr/share/neo4j/lib/* -XX:+UseParallelGC -XX:-OmitStackTraceInFastThrow -XX:+UnlockExperimentalVMOptions -XX:+TrustFinalNonStaticFields -XX:+DisableExplicitGC -Djdk.nio.maxCachedBufferSize=1024 -Dio.netty.tryReflectionSetAccessible=true -XX:+ExitOnOutOfMemoryError -Djdk.tls.ephemeralDHKeySize=2048 -XX:FlightRecorderOptions=stackdepth=256 -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED -Dlog4j2.disable.jmx=true -Xlog:gc*,safepoint,age*=trace:file=/var/log/neo4j/gc.log::filecount=5,filesize=20480k -Dfile.encoding=UTF-8 org.neo4j.cli.AdminTool database dump --verbose --to-path=/home/shaofu/backup_neo4j_samples neo4j
neo4j 5.22.0
VM Name: OpenJDK 64-Bit Server VM
VM Vendor: Ubuntu
VM Version: 21.0.4+7-Ubuntu-1ubuntu222.04
JIT compiler: HotSpot 64-Bit Tiered Compilers
VM Arguments: [-XX:+UseParallelGC, -XX:-OmitStackTraceInFastThrow, -XX:+UnlockExperimentalVMOptions, -XX:+TrustFinalNonStaticFields, -XX:+DisableExplicitGC, -Djdk.nio.maxCachedBufferSize=1024, -Dio.netty.tryReflectionSetAccessible=true, -XX:+ExitOnOutOfMemoryError, -Djdk.tls.ephemeralDHKeySize=2048, -XX:FlightRecorderOptions=stackdepth=256, -XX:+UnlockDiagnosticVMOptions, -XX:+DebugNonSafepoints, --add-opens=java.base/java.nio=ALL-UNNAMED, --add-opens=java.base/java.io=ALL-UNNAMED, --add-opens=java.base/sun.nio.ch=ALL-UNNAMED, -Dlog4j2.disable.jmx=true, -Xlog:gc*,safepoint,age*=trace:file=/var/log/neo4j/gc.log::filecount=5,filesize=20480k, -Dfile.encoding=UTF-8]
Configuration files used (ordered by priority):
/etc/neo4j/neo4j-admin.conf
/etc/neo4j/neo4j.conf
--------------------
2024-11-06 01:25:38.338+0000 INFO  [o.n.c.d.DumpCommand] Starting dump of database 'neo4j'
Done: 38 files, 556.2MiB processed.
2024-11-06 01:25:47.104+0000 INFO  [o.n.c.d.DumpCommand] Dump completed successfully
```

```shell
期望将两个neo4j的数据进行合并，而不是使用dump文件进行替代

TODO 这个基本上不可行的样子，社区版本的command没有对应的export指令

neo4j-admin database export --database=neo4j --nodes=/data/nodes.csv --relationships=/data/relationships.csv
neo4j-admin export --database=neo4j --nodes=/data/nodes.csv --relationships=/data/relationships.csv
```

```shell
将上面导出的dump文件加载到当前的neo4j中（全量替代-有不足的，更好的是使用部分内容的加载，这样能够进行数据的合并）（注意sudo）

shaofu@shaofu-Aspire-4741:~/backup_of_neo4j$ pwd
/home/shaofu/backup_of_neo4j
shaofu@shaofu-Aspire-4741:~/backup_of_neo4j$ tree .
.
└── neo4j.dump

0 directories, 1 file

sudo neo4j-admin database load --from-path=/home/shaofu/backup_of_neo4j neo4j --verbose

sudo neo4j-admin database load --from-path=/data neo4j --verbose

Tips：新版本的neo4j的指令似乎不一样了，和ai给出的指令差异比较大
```

安装neo4j的时候需要增加源source：https://debian.neo4j.com/ （应该是发生了比较大的变化，ai给出的内容已经不够准确，参考官方的现有文档的内容才行）

```shell
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
```

```shell
root@0df62351e774:/data# sudo neo4j-admin database load --from-path=/data neo4j --verbose
bash: sudo: command not found
root@0df62351e774:/data# neo4j-admin database load --from-path=/data neo4j --verbose
Executing command line: /opt/java/openjdk/bin/java -cp /var/lib/neo4j/plugins/*:/var/lib/neo4j/conf/*:/var/lib/neo4j/lib/* -XX:+UseParallelGC -XX:-OmitStackTraceInFastThrow -XX:+UnlockExperimentalVMOptions -XX:+TrustFinalNonStaticFields -XX:+DisableExplicitGC -Djdk.nio.maxCachedBufferSize=1024 -Dio.netty.tryReflectionSetAccessible=true -XX:+ExitOnOutOfMemoryError -Djdk.tls.ephemeralDHKeySize=2048 -XX:FlightRecorderOptions=stackdepth=256 -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -Dlog4j2.disable.jmx=true -Dfile.encoding=UTF-8 org.neo4j.cli.AdminTool database load --from-path=/data neo4j --verbose
neo4j 5.25.1
VM Name: OpenJDK 64-Bit Server VM
VM Vendor: Eclipse Adoptium
VM Version: 17.0.13+11
JIT compiler: HotSpot 64-Bit Tiered Compilers
VM Arguments: [-XX:+UseParallelGC, -XX:-OmitStackTraceInFastThrow, -XX:+UnlockExperimentalVMOptions, -XX:+TrustFinalNonStaticFields, -XX:+DisableExplicitGC, -Djdk.nio.maxCachedBufferSize=1024, -Dio.netty.tryReflectionSetAccessible=true, -XX:+ExitOnOutOfMemoryError, -Djdk.tls.ephemeralDHKeySize=2048, -XX:FlightRecorderOptions=stackdepth=256, -XX:+UnlockDiagnosticVMOptions, -XX:+DebugNonSafepoints, --add-opens=java.base/java.nio=ALL-UNNAMED, --add-opens=java.base/java.io=ALL-UNNAMED, --add-opens=java.base/sun.nio.ch=ALL-UNNAMED, --enable-native-access=ALL-UNNAMED, -Dlog4j2.disable.jmx=true, -Dfile.encoding=UTF-8]
Configuration files used (ordered by priority):
/var/lib/neo4j/conf/neo4j-admin.conf
/var/lib/neo4j/conf/neo4j.conf
--------------------
Failed to load database 'neo4j': The database is in use. Stop database 'neo4j' and try again.
Load failed for databases: 'neo4j'
org.neo4j.cli.CommandFailedException: Load failed for databases: 'neo4j'
	at org.neo4j.commandline.dbms.LoadCommand.checkFailure(LoadCommand.java:305)
	at org.neo4j.commandline.dbms.LoadCommand.loadDump(LoadCommand.java:288)
	at org.neo4j.commandline.dbms.LoadCommand.loadDump(LoadCommand.java:246)
	at org.neo4j.commandline.dbms.LoadCommand.execute(LoadCommand.java:174)
	at org.neo4j.cli.AbstractCommand.call(AbstractCommand.java:92)
	at org.neo4j.cli.AbstractCommand.call(AbstractCommand.java:37)
	at picocli.CommandLine.executeUserObject(CommandLine.java:2045)
	at picocli.CommandLine.access$1500(CommandLine.java:148)
	at picocli.CommandLine$RunLast.executeUserObjectOfLastSubcommandWithSameParent(CommandLine.java:2465)
	at picocli.CommandLine$RunLast.handle(CommandLine.java:2457)
	at picocli.CommandLine$RunLast.handle(CommandLine.java:2419)
	at picocli.CommandLine$AbstractParseResultHandler.execute(CommandLine.java:2277)
	at picocli.CommandLine$RunLast.execute(CommandLine.java:2421)
	at picocli.CommandLine.execute(CommandLine.java:2174)
	at org.neo4j.cli.AdminTool.execute(AdminTool.java:94)
	at org.neo4j.cli.AdminTool.main(AdminTool.java:82)
Caused by: org.neo4j.cli.CommandFailedException: The database is in use. Stop database 'neo4j' and try again.
	at org.neo4j.commandline.dbms.LoadDumpExecutor.execute(LoadDumpExecutor.java:82)
	at org.neo4j.commandline.dbms.LoadCommand.loadDump(LoadCommand.java:279)
	... 14 more
Caused by: org.neo4j.io.locker.FileLockException: Lock file has been locked by another process: /data/databases/neo4j/database_lock. Please ensure no other process is using this database, and that the directory is writable (required even for read-only access)
	at org.neo4j.io.locker.Locker.storeLockException(Locker.java:151)
	at org.neo4j.io.locker.Locker.checkLock(Locker.java:82)
	at org.neo4j.kernel.internal.locker.GlobalFileLocker.checkLock(GlobalFileLocker.java:55)
	at org.neo4j.kernel.internal.locker.DatabaseLocker.checkLock(DatabaseLocker.java:28)
	at org.neo4j.commandline.dbms.LockChecker.checkLock(LockChecker.java:81)
	at org.neo4j.commandline.dbms.LockChecker.check(LockChecker.java:65)
	at org.neo4j.commandline.dbms.LockChecker.checkDatabaseLock(LockChecker.java:53)
	at org.neo4j.commandline.dbms.LoadDumpExecutor.execute(LoadDumpExecutor.java:78)
	... 15 more
```


```shell
想要在docker启动的neo4j中将dummp文件导入，但是失败了，因为容器就是启动了neo4j，但是导入却是需要将neo4j的进程停止掉（停机导入），矛盾了

sudo docker exec -it 0df62351e774843bd9631b985709075954603128f33d05829f245a26b915c643 neo4j-admin database load --from-path=/data/bakup neo4j --verbose
```

```shell
 pip install fastapi
 pip install flask_sqlalchemy
 pip install py2neo
 pip install pymysql
 pip install elasticsearch
 pip install pyformlang
 pip install flask_socketio
 
```



neo4j-admin load [--expand-commands] [--force] [--info] [--verbose]
                 [--database=<database>] --from=<path>


				 