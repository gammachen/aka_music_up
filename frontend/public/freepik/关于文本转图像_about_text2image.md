## 文生图在线endpoint

```shell
https://docs.freepik.com/api-reference/mystic/post-mystic
```

```shell
一个简单的文生图提示词：

可爱的小女孩，白头发，猫耳朵，全身照，杰作，最好的画质，1个女孩，令人惊叹，美丽的细节眼睛，精细地细节，景深，非常详细的CG，原创的，非常详细的墙纸，上半身，看着观众，an anime drawing，动画风格

Cute little girl, white hair, cat ears, full body shot, masterpiece, best quality, 1 girl, stunning, beautiful detailed eyes, fine details, depth of field, highly detailed CG, original, highly detailed wallpaper, upper body, looking at viewer, an anime drawing, anime style
```

```shell
curl --request POST \
  --url https://api.freepik.com/v1/ai/mystic \
  --header 'Content-Type: application/json' \
  --header 'x-freepik-api-key: <api-key>' \
  --data '{
  "prompt": "<string>",
  "webhook_url": "https://httpbin.org/post",
  "structure_reference": "aSDinaTvuI8gbWludGxpZnk=",
  "structure_strength": 50,
  "style_reference": "aSDinaTvuI8gbWludGxpZnk=",
  "adherence": 50,
  "hdr": 50,
  "resolution": "2k",
  "aspect_ratio": "square_1_1",
  "realism": true,
  "creative_detailing": 33,
  "engine": "automatic",
  "fixed_generation": false,
  "filter_nsfw": true,
  "styling": {
    "styles": [
      {
        "name": "<string>",
        "strength": 100
      }
    ],
    "characters": [
      {
        "id": "<string>",
        "strength": 100
      }
    ]
  }
}'
```

```shell
curl --request POST \
  --url https://api.freepik.com/v1/ai/mystic \
  --header 'Content-Type: application/json' \
  --header 'x-freepik-api-key: FPSX031fc90b071a4a50b5cd6e8af596ab74' \
  --data '{
  "prompt": "Cute little girl, white hair, cat ears, full body shot, masterpiece, best quality, 1 girl, stunning, beautiful detailed eyes, fine details, depth of field, highly detailed CG, original, highly detailed wallpaper, upper body, looking at viewer, an anime drawing, anime style",
  "webhook_url": "https://alphago.ltd/freepik/taskcallback",
  "adherence": 50,
  "hdr": 50,
  "resolution": "2k",
  "aspect_ratio": "square_1_1",
  "realism": true,
  "creative_detailing": 33,
  "engine": "automatic",
  "fixed_generation": false,
  "filter_nsfw": true
}'

{"data":{"task_id":"3bcb9272-f256-474a-95c6-d67c7a883739","status":"CREATED","generated":[]}}%

curl --request GET \
  --url https://api.freepik.com/v1/ai/mystic/3bcb9272-f256-474a-95c6-d67c7a883739 \
  --header 'x-freepik-api-key: FPSX031fc90b071a4a50b5cd6e8af596ab74'

{"data":{"task_id":"3bcb9272-f256-474a-95c6-d67c7a883739","status":"COMPLETED","generated":["https://ai-statics.freepik.com/content/mg-upscaler/4zmh4g3dvrgzhp2xxchgzwuon4/output.png?token=exp=1743228169~hmac=e7950c25464eeb43ffb598eb45deb718"],"has_nsfw":[false]}}


```

```shell
波普艺术与赛博朋克风格融合的数字艺术作品，画面中央是一位穿着未来感银色紧身衣的女性，她的头发被设计成鲜艳的霓虹绿色，眼睛则被替换成了机械义眼，闪烁着冷酷的蓝光。背景是一座充满赛博朋克元素的城市，高楼大厦的霓虹灯与广告牌交织出一片光怪陆离的夜景。街道上行人匆匆，全息投影与无人机在空中穿梭。整个画面采用印象派的手法处理，色彩斑斓，光影交错，营造出一种既科幻又具有艺术气息的视觉效果。近景与中景的对比强烈，展现了科技与人性的冲突与融合。

A digital artwork that fuses Pop Art and Cyberpunk styles, featuring a woman in a futuristic silver bodysuit at the center of the image. Her hair is designed in vibrant neon green, and her eyes are replaced with mechanical cybernetic eyes that emit a cold blue light. The background depicts a cyberpunk city where neon lights from skyscrapers interweave with billboards, creating a bizarre nightscape. People hurry along the streets while holograms and drones traverse the sky. The entire image is rendered in an impressionistic style, with rich colors and interplaying light and shadows, creating a visual effect that is both sci-fi and artistic. The strong contrast between the foreground and middle ground showcases the conflict and fusion between technology and humanity.


curl --request POST \
  --url https://api.freepik.com/v1/ai/mystic \
  --header 'Content-Type: application/json' \
  --header 'x-freepik-api-key: FPSX031fc90b071a4a50b5cd6e8af596ab74' \
  --data '{
  "prompt": "A digital artwork that fuses Pop Art and Cyberpunk styles, featuring a woman in a futuristic silver bodysuit at the center of the image. Her hair is designed in vibrant neon green, and her eyes are replaced with mechanical cybernetic eyes that emit a cold blue light. The background depicts a cyberpunk city where neon lights from skyscrapers interweave with billboards, creating a bizarre nightscape. People hurry along the streets while holograms and drones traverse the sky. The entire image is rendered in an impressionistic style, with rich colors and interplaying light and shadows, creating a visual effect that is both sci-fi and artistic. The strong contrast between the foreground and middle ground showcases the conflict and fusion between technology and humanity.",
  "webhook_url": "https://alphago.ltd/freepik/taskcallback",
  "adherence": 50,
  "hdr": 50,
  "resolution": "2k",
  "aspect_ratio": "square_1_1",
  "realism": true,
  "creative_detailing": 33,
  "engine": "automatic",
  "fixed_generation": false,
  "filter_nsfw": true
}'

{"data":{"task_id":"12a03b62-32d6-425e-830e-5747f4c01f5b","status":"CREATED","generated":[]}}

curl --request GET \
  --url https://api.freepik.com/v1/ai/mystic/12a03b62-32d6-425e-830e-5747f4c01f5b \
  --header 'x-freepik-api-key: FPSX031fc90b071a4a50b5cd6e8af596ab74'

{"data":{"task_id":"12a03b62-32d6-425e-830e-5747f4c01f5b","status":"COMPLETED","generated":["https://ai-statics.freepik.com/content/mg-upscaler/hzjpt2t2w5f3pk22lodhaxqdwm/output.png?token=exp=1743231244~hmac=54094dc22bd57e9e2e487a5746331811"],"has_nsfw":[false]}}

wget 'https://ai-statics.freepik.com/content/mg-upscaler/hzjpt2t2w5f3pk22lodhaxqdwm/output.png?token=exp=1743231244~hmac=54094dc22bd57e9e2e487a5746331811'
```