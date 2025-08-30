curl -X POST http://localhost:3002/v1/crawl \
    -H 'Content-Type: application/json' \
    -d '{
      "url": "https://www.xiaohuasheng.cn/roundtable/32012"
    }'
{"success":true,"id":"f1e031b5-d1df-48f7-a764-4f1a64606d8c","url":"https://localhost:3002/v1/crawl/f1e031b5-d1df-48f7-a764-4f1a64606d8c"}%

(base) shhaofu@shhaofudeMacBook-Pro ScreenCapture % curl -X GET http://localhost:3002/v1/crawl/f1e031b5-d1df-48f7-a764-4f1a64606d8c \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer localapikey'