from manim import *
from manim.utils.tex_templates import TexTemplate

class ChainRuleDemo(Scene):
    def construct(self):
        # 配置中文公式模板
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble=r'\usepackage{amsmath}\usepackage{ctex}\usepackage{fontspec}\setmainfont{STHeiti}\setromanfont{STHeiti}\setsansfont{STHeiti}\setmonofont{Menlo}\usepackage{xeCJK}\setCJKmainfont{STHeiti}\setCJKsansfont{STHeiti}\setCJKmonofont{STHeiti}'
        )

        # 创建标题和公式
        title = Tex(r"\text{链式法则演示}", tex_template=ctex_template, font_size=40).to_edge(UP)
        
        equation = VGroup(
            Tex(r"$f(x) = \sin(x^2)$", tex_template=ctex_template),
            Tex(r"$f'(x) = \cos(x^2) \cdot 2x$", tex_template=ctex_template,
                color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).scale(0.9).to_edge(LEFT)

        # 创建坐标系和函数曲线
        axes = Axes(x_range=[-3,3], y_range=[-2,2])
        composite_graph = axes.plot(lambda x: np.sin(x**2), color=BLUE)
        derivative_graph = axes.plot(lambda x: np.cos(x**2)*2*x, color=RED)

        # 导数分解可视化元素
        inner_func = axes.plot(lambda x: x**2, color=GREEN)
        outer_func = axes.plot(lambda x: np.sin(x), color=ORANGE)
        
        # 动画流程
        self.play(Write(title))
        self.play(Create(axes), Create(composite_graph))
        self.wait()

        # 显示链式法则公式
        self.play(Write(equation[0]))
        self.wait()

        # 分解导数过程
        decomposition = VGroup(
            Tex(r"外层: $\sin(u)$", tex_template=ctex_template, color=ORANGE),
            Tex(r"内层: $u = x^2$", tex_template=ctex_template, color=GREEN),
            Tex(r"外层导数: $\cos(u)$", tex_template=ctex_template, color=ORANGE),
            Tex(r"内层导数: $2x$", tex_template=ctex_template, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.7).to_edge(RIGHT)

        # 导数分解动画
        self.play(Transform(composite_graph, inner_func),
                  Write(decomposition[1]))
        self.wait()

        self.play(Transform(composite_graph, outer_func),
                  Write(decomposition[0]))
        self.wait()

        # 显示完整导数公式
        self.play(Write(equation[1]),
                  Create(derivative_graph))
        
        # 添加连接箭头
        arrow_group = VGroup(
            Arrow(decomposition[0].get_bottom(), decomposition[2].get_top(), color=ORANGE),
            Arrow(decomposition[1].get_bottom(), decomposition[3].get_top(), color=GREEN),
            Arrow(equation[1].get_left(), axes.get_right(), color=YELLOW)
        )
        self.play(*[GrowArrow(a) for a in arrow_group])
        self.wait(2)

# 运行命令：manim -pql chain_rule_demo.py ChainRuleDemo