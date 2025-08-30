from manim import *
from manim.utils.tex_templates import TexTemplate

class DerivativeAnimation(Scene):
    def construct(self):
        # 配置中文公式模板
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble='\\usepackage{amsmath}\\usepackage{ctex}\\usepackage{fontspec}\\setmainfont{STHeiti}\\setromanfont{STHeiti}\\setsansfont{STHeiti}\\setmonofont{Menlo}\\usepackage{xeCJK}\\setCJKmainfont{STHeiti}\\setCJKsansfont{STHeiti}\\setCJKmonofont{STHeiti}'
        )
        
        # 创建公式组
        title = Tex(r"\text{微分基本定理}", tex_template=ctex_template, font_size=40).to_edge(UP)
        
        equations = VGroup(
            Tex(r"1. 一阶导数：$f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$", tex_template=ctex_template),
            Tex(r"2. 常数：$C'=0$", tex_template=ctex_template),
            Tex(r"3. Cf：$(Cf)'=Cf'$", tex_template=ctex_template),
            Tex(r"4. f+g：$(f+g)'=f'+g'$", tex_template=ctex_template),
            Tex(r"5. 次方：$(x^n)'=nx^{n-1}$", tex_template=ctex_template),
            Tex(r"6. 乘积：$(fg)'=f'g+fg'$", tex_template=ctex_template),
            Tex(r"7. 商：$(\frac{f}{g})'=\frac{f'g-fg'}{g^2}$", tex_template=ctex_template),
            Tex(r"8. 链式法则：$\frac{d}{dx}f(g(x))=f'(g(x))g'(x)$", tex_template=ctex_template)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.7).to_edge(LEFT)

        # 创建坐标系和函数曲线
        axes = Axes(x_range=[-3,3], y_range=[-5,5])
        func_graph = axes.plot(lambda x: x**2, color=BLUE)
        deriv_graph = axes.plot(lambda x: 2*x, color=RED)

        # 初始动画
        self.play(Write(title))
        self.play(Create(axes), Create(func_graph))
        self.wait(1)

        # 分步展示
        for i, eq in enumerate(equations):
            self.play(Write(eq))
            self.animate_derivative(i+1, axes, func_graph, deriv_graph)
            self.wait(1)

    def animate_derivative(self, step, axes, f_graph, d_graph):
        if step == 1:
            # 展示极限过程动画
            tangent_line = axes.get_secant_slope_group(
                x=1, graph=f_graph, dx=0.1,
                secant_line_color=GREEN,
                secant_line_length=4)
            self.play(Create(tangent_line))
        elif step == 5:
            # 幂函数导数动态演示
            self.play(Transform(
                f_graph,
                axes.plot(lambda x: x**3, color=BLUE)
            ))
            self.play(Transform(
                d_graph,
                axes.plot(lambda x: 3*x**2, color=RED)
            ))
        elif step == 8:
            # 链式法则路径追踪
            composite_graph = axes.plot(
                lambda x: np.sin(x**2), color=YELLOW)
            self.play(Create(composite_graph))

# 运行命令:
# manim -pql derivative_animation.py DerivativeAnimation