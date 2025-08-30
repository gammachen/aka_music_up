
```shell
搜索优化器的查询成本
mysql> select count(*) from A;
+----------+
| count(*) |
+----------+
|     9744 |
+----------+
1 row in set (0.01 sec)

mysql> show status like 'Last_query_cost';
+-----------------+------------+
| Variable_name   | Value      |
+-----------------+------------+
| Last_query_cost | 976.999000 |
+-----------------+------------+
1 row in set (0.00 sec)

mysql> select count(*) from B;
+----------+
| count(*) |
+----------+
|    10000 |
+----------+
1 row in set (0.00 sec)

mysql> show status like 'Last_query_cost';
+-----------------+------------+
| Variable_name   | Value      |
+-----------------+------------+
| Last_query_cost | 987.449000 |
+-----------------+------------+
1 row in set (0.00 sec)

mysql> select * from B limit 100000000,1;
Empty set (0.03 sec)

mysql> show status like 'Last_query_cost';
+-----------------+------------+
| Variable_name   | Value      |
+-----------------+------------+
| Last_query_cost | 987.449000 |
+-----------------+------------+
1 row in set (0.02 sec)

mysql> select * from B limit 100000000,10000;
Empty set (0.01 sec)

mysql> show status like 'Last_query_cost';
+-----------------+------------+
| Variable_name   | Value      |
+-----------------+------------+
| Last_query_cost | 987.449000 |
+-----------------+------------+
1 row in set (0.01 sec)
```



