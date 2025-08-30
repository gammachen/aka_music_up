from manim import *
from manim.utils.tex_templates import TexTemplate
from sympy import *

class ProductRuleAnimation(Scene):
    def construct(self):
        # 配置中文公式模板
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble='\\usepackage{amsmath}\\usepackage{ctex}\\usepackage{fontspec}\\setmainfont{STHeiti}\\setromanfont{STHeiti}\\setsansfont{STHeiti}\\setmonofont{Menlo}\\usepackage{xeCJK}\\setCJKmainfont{STHeiti}\\setCJKsansfont{STHeiti}\\setCJKmonofont{STHeiti}'
        )
        
        # 创建标题
        title = Tex(r"\text{乘积求导法则演示}", tex_template=ctex_template, font_size=40).to_edge(UP)
        self.play(Write(title))
        
        # 创建函数定义
        function_def = VGroup(
            Tex(r"$f(x) = x^2$", tex_template=ctex_template, color=BLUE),
            Tex(r"$g(x) = x^3$", tex_template=ctex_template, color=RED),
            Tex(r"$h(x) = f(x) \cdot g(x) = x^2 \cdot x^3 = x^5$", tex_template=ctex_template, color=PURPLE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8).to_edge(LEFT)
        
        # 显示函数定义
        self.play(Write(function_def[0]))
        self.play(Write(function_def[1]))
        self.play(Write(function_def[2]))
        self.wait(1)
        
        # 创建乘积法则公式
        product_rule = Tex(r"$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$", 
                           tex_template=ctex_template, color=YELLOW).scale(0.9)
        product_rule.next_to(function_def, DOWN, buff=0.8)
        
        # 显示乘积法则
        self.play(Write(product_rule))
        self.wait(1)
        
        # 创建坐标系和函数图像
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-5, 30, 5],
            axis_config={"include_tip": True},
        ).scale(0.6).to_edge(RIGHT)
        
        # 添加坐标轴标签
        x_label = Tex("x", tex_template=ctex_template).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Tex("y", tex_template=ctex_template).next_to(axes.y_axis.get_end(), UP)
        axes_labels = VGroup(x_label, y_label)
        
        # 创建函数图像
        f_graph = axes.plot(lambda x: x**2, color=BLUE)
        g_graph = axes.plot(lambda x: x**3, color=RED)
        h_graph = axes.plot(lambda x: x**5, color=PURPLE)
        
        # 显示坐标系和函数图像
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(f_graph))
        self.play(Create(g_graph))
        self.play(Create(h_graph))
        self.wait(1)
        
        # 计算导数
        calculation = VGroup(
            Tex(r"1. 计算 $f'(x)$:", tex_template=ctex_template),
            Tex(r"$f'(x) = \frac{d}{dx}(x^2) = 2x$", tex_template=ctex_template, color=BLUE),
            Tex(r"2. 计算 $g'(x)$:", tex_template=ctex_template),
            Tex(r"$g'(x) = \frac{d}{dx}(x^3) = 3x^2$", tex_template=ctex_template, color=RED),
            Tex(r"3. 应用乘积法则:", tex_template=ctex_template),
            Tex(r"$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$", tex_template=ctex_template, color=YELLOW),
            Tex(r"$= 2x \cdot x^3 + x^2 \cdot 3x^2$", tex_template=ctex_template),
            Tex(r"$= 2x^4 + 3x^4$", tex_template=ctex_template),
            Tex(r"$= 5x^4$", tex_template=ctex_template, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.7)
        
        # 重新定位计算过程
        calculation.next_to(product_rule, DOWN, buff=0.8)
        
        # 逐步显示计算过程
        for i, step in enumerate(calculation):
            self.play(Write(step))
            if i == 1 or i == 3 or i == 8:  # 在关键步骤后暂停
                self.wait(1)
        
        # 显示最终导数图像
        derivative_graph = axes.plot(lambda x: 5*x**4, color=GREEN)
        self.play(Create(derivative_graph))
        
        # 使用SymPy验证结果
        sympy_verification = VGroup(
            Tex(r"使用SymPy验证:", tex_template=ctex_template),
            Tex(r"$x = Symbol('x')$", tex_template=ctex_template),
            Tex(r"$f(x) = x^2$", tex_template=ctex_template),
            Tex(r"$g(x) = x^3$", tex_template=ctex_template),
            Tex(r"$h(x) = f(x) \cdot g(x)$", tex_template=ctex_template),
            Tex(r"$h'(x) = \frac{d}{dx}[h(x)]$", tex_template=ctex_template),
            Tex(r"$h'(x) = 5x^4$", tex_template=ctex_template, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.7)
        
        # 将SymPy验证移到屏幕右下角
        sympy_verification.to_edge(DOWN).shift(UP*0.5)
        
        # 显示SymPy验证
        self.play(FadeOut(calculation))
        for step in sympy_verification:
            self.play(Write(step))
        
        # 最终总结
        conclusion = Tex(r"乘积求导法则: $(f \cdot g)' = f' \cdot g + f \cdot g'$", 
                         tex_template=ctex_template, color=YELLOW).scale(1.2)
        conclusion.to_edge(DOWN)
        
        self.play(FadeOut(sympy_verification))
        self.play(Write(conclusion))
        self.wait(2)

# 运行命令:
# manim -pql product_rule_animation.py ProductRuleAnimation