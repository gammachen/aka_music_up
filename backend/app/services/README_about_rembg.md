


u2net : 一个适用于通用场景的预训练模型。
u2netp : u2net 模型的轻量级版本。
u2net_human_seg : 一个用于人体分割的预训练模型。
u2net_cloth_seg : 一个从人像中解析衣物的预训练模型。衣物被分为三类：上身、下身和全身。
silueta : 与 u2net 相同，但大小减少到 43Mb。
isnet-general-use : 一个新的适用于通用场景的预训练模型。
isnet-anime : 一个针对动漫角色的高精度分割模型。
sam : 一个适用于任何场景的预训练模型。
birefnet-general : 一个适用于通用场景的预训练模型。
birefnet-general-lite : 一个适用于通用场景的轻量级预训练模型。
birefnet-portrait : 一个用于人像的预训练模型。
birefnet-dis : 一个用于二值图像分割（DIS）的预训练模型。
birefnet-hrsod : 一个用于高分辨率显著物体检测（HRSOD）的预训练模型。
birefnet-cod : 一个用于隐匿物体检测（COD）的预训练模型。
birefnet-massive : 一个基于大规模数据集训练的预训练模型


---- 这个似乎没什么意义啊
ffmpeg -i chang_an_san_wangli.mp4 -ss 10 -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b 1280 720 -o folder/chang_an_san_wangli-%03u.png