
```shell
docker compose -f docker-compose-image-tag.yml up
```

```shell
superset_init         | /app/superset/config.py:42: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
superset_init         |   import pkg_resources
superset_init         | Loaded your LOCAL configuration at [/app/docker/pythonpath_dev/superset_config.py]
superset_init         | 2025-06-21 16:34:59,673:DEBUG:superset.utils.logging_configurator:logging was configured successfully
```

```shell
docker的yml必须修改，否则下载不来镜像

#x-superset-image: &superset-image apachesuperset.docker.scarf.sh/apache/superset:${TAG:-latest-dev}
# shaofu modified 06-22
x-superset-image: &superset-image apache/superset:${TAG:-latest-dev}

```


```shell
链接mysql必须开启非安全链接的先：

PREVENT_UNSAFE_DB_CONNECTIONS = False
```