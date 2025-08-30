## 将mobi文件转成PDF

```shell
(base) shhaofu@shhaofudeMacBook-Pro aka_music % ebook-convert
用法: ebook-convert input_file output_file [options]

转换不同格式的电子书。

input_file 表示输入文件，output_file 表示输出文件。这两者作为命令行参数必须指定到最前面。

输出的电子书格式可由 output_file 的扩展名得到。同时 output_file 也可以是一种以 .EXT 为扩展名的特殊格式。在这种情况下，输出文件的名称则使用输入文件的名称。注意：文件名不能以连字号作为开头。如果 output_file 不含扩展名，那么它将被视为一个目录并将会在该目录下生成 HTML 格式的“开放式电子书(OEB)”。这些文件会被视为正常文件而被输出插件所识别。

在指定输入和输出文件后，你可以自定义特定的转换选项。根据输入和输出文件的类型不同可用的转换选项也不同。如需获取针对输入和输出文件的帮助，请在命令行中输入 -h。

对于转换系统的完整文档请查阅
https://manual.calibre-ebook.com/conversion.html

每当向具有它们自己空间的ebook-convert传递参数时，用引号括起这些参数。例如: "/some path/with spaces"

選項:
  --version       顯示程式版本編號並退出

  -h, --help      顯示說明訊息並退出

  --list-recipes  列出内建的订阅清单名。你可以通过如下命令创建基于内建订阅清单的电子书： ebook-convert "Recipe
                  Name.recipe" output.epub


建立者：Kovid Goyal <kovid@kovidgoyal.net>

```

