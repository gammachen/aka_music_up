
```bash
cluster.name: "docker-cluster"
network.host: 0.0.0.0

#----------------------- BEGIN SECURITY AUTO CONFIGURATION -----------------------
#
# The following settings, TLS certificates, and keys have been automatically
# generated to configure Elasticsearch security features on 09-04-2025 16:55:15
#
# --------------------------------------------------------------------------------

# Enable security features
xpack.security.enabled: true

xpack.security.enrollment.enabled: true

# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents
xpack.security.http.ssl:
  enabled: true
  keystore.path: certs/http.p12

# Enable encryption and mutual authentication between cluster nodes
xpack.security.transport.ssl:
  enabled: true
  verification_mode: certificate
  keystore.path: certs/transport.p12
  truststore.path: certs/transport.p12
#----------------------- END SECURITY AUTO CONFIGURATION -------------------------
cluster.routing.allocation.disk.watermark.low: "95%"
cluster.routing.allocation.disk.watermark.high: "97%"
cluster.routing.allocation.disk.watermark.flood_stage: "98%"
```

```bash
[2025-04-10T03:45:20,987][WARN ][logstash.licensechecker.licensereader] Attempted to resurrect connection to dead ES instance, but got an error {:url=>"http://elasticsearch:9200/", :exception=>LogStash::Outputs::ElasticSearch::HttpClient::Pool::HostUnreachableError, :message=>"Elasticsearch Unreachable: [http://elasticsearch:9200/][Manticore::ResolutionFailure] elasticsearch: Name or service not known"}
[2025-04-10T03:45:50,511][ERROR][logstash.licensechecker.licensereader] Unable to retrieve license information from license server {:message=>"No Available connections"}
[2025-04-10T03:45:51,091][INFO ][logstash.licensechecker.licensereader] Failed to perform request {:message=>"elasticsearch: Name or service not known", :exception=>Manticore::ResolutionFailure, :cause=>#<Java::JavaNet::UnknownHostException: elasticsearch: Name or service not known>}
[2025-04-10T03:45:51,092][WARN ][logstash.licensechecker.licensereader] Attempted to resurrect connection to dead ES instance, but got an error {:url=>"http://elasticsearch:9200/", :exception=>LogStash::Outputs::ElasticSearch::HttpClient::Pool::HostUnreachableError, :message=>"Elasticsearch Unreachable: [http://elasticsearch:9200/][Manticore::ResolutionFailure] elasticsearch: Name or service not known"}
```

cat <<EOF > elasticsearch.yml
cluster.name: "docker-cluster"
network.host: 0.0.0.0

#----------------------- BEGIN SECURITY AUTO CONFIGURATION -----------------------
#
# The following settings, TLS certificates, and keys have been automatically
# generated to configure Elasticsearch security features on 09-04-2025 16:55:15
#
# --------------------------------------------------------------------------------

# Enable security features
xpack.security.enabled: false

xpack.security.enrollment.enabled: false

# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents
xpack.security.http.ssl:
  enabled: false
  keystore.path: certs/http.p12

# Enable encryption and mutual authentication between cluster nodes
xpack.security.transport.ssl:
  enabled: false
  verification_mode: certificate
  keystore.path: certs/transport.p12
  truststore.path: certs/transport.p12
#----------------------- END SECURITY AUTO CONFIGURATION -------------------------
cluster.routing.allocation.disk.watermark.low: "95%"
cluster.routing.allocation.disk.watermark.high: "97%"
cluster.routing.allocation.disk.watermark.flood_stage: "98%"
EOF