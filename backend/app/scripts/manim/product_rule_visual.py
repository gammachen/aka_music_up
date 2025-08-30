from manim import *
from manim.utils.tex_templates import TexTemplate
from sympy import *

class ProductRuleVisual(Scene):
    def construct(self):
        # 配置中文公式模板
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble='\\usepackage{amsmath}\\usepackage{ctex}\\usepackage{fontspec}\\setmainfont{STHeiti}\\setromanfont{STHeiti}\\setsansfont{STHeiti}\\setmonofont{Menlo}\\usepackage{xeCJK}\\setCJKmainfont{STHeiti}\\setCJKsansfont{STHeiti}\\setCJKmonofont{STHeiti}'
        )
        
        # 创建标题 - 减小字体并紧贴顶部
        title = Tex(r"\text{乘积求导法则可视化}", tex_template=ctex_template, font_size=36).to_edge(UP, buff=0.1)
        self.play(Write(title))
        self.wait(0.5)
        
        # 创建问题描述 - 减小字体和间距
        problem = Tex(r"\text{计算} $\frac{d}{dx}[f(x)g(x)]$, \text{其中} $f(x)=x^2, g(x)=x^3$", 
                      tex_template=ctex_template).scale(0.7).next_to(title, DOWN, buff=0.2)
        self.play(Write(problem))
        self.wait(1)
        
        # 创建函数定义和图像区域 - 调整位置和大小
        left_area = Rectangle(height=6, width=6, color=WHITE, fill_opacity=0).to_edge(LEFT, buff=0.3).shift(DOWN*0.2)
        right_area = Rectangle(height=6, width=6, color=WHITE, fill_opacity=0).to_edge(RIGHT, buff=0.3).shift(DOWN*0.2)
        
        # 函数定义 - 减小字体和间距
        function_def = VGroup(
            Tex(r"\text{步骤1: 定义函数}", tex_template=ctex_template, color=YELLOW),
            Tex(r"$f(x) = x^2$", tex_template=ctex_template, color=BLUE),
            Tex(r"$g(x) = x^3$", tex_template=ctex_template, color=RED),
            Tex(r"$h(x) = f(x) \cdot g(x) = x^2 \cdot x^3 = x^5$", tex_template=ctex_template, color=PURPLE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).scale(0.6)
        function_def.move_to(left_area.get_center() + UP*1.8)
        
        # 显示函数定义
        self.play(Create(left_area), Create(right_area))
        for item in function_def:
            self.play(Write(item))
        self.wait(1)
        
        # 创建坐标系和函数图像 - 减小大小并调整位置
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-5, 30, 5],
            axis_config={"include_tip": True},
        ).scale(0.45).move_to(right_area.get_center())
        
        # 添加坐标轴标签 - 减小标签
        x_label = Tex("x", tex_template=ctex_template).next_to(axes.x_axis.get_end(), RIGHT, buff=0.05).scale(0.6)
        y_label = Tex("y", tex_template=ctex_template).next_to(axes.y_axis.get_end(), UP, buff=0.05).scale(0.6)
        axes_labels = VGroup(x_label, y_label)
        
        # 创建函数图像
        f_graph = axes.plot(lambda x: x**2, color=BLUE)
        g_graph = axes.plot(lambda x: x**3, color=RED)
        h_graph = axes.plot(lambda x: x**5, color=PURPLE)
        
        # 函数图例 - 减小大小并调整位置
        legend = VGroup(
            Tex(r"$f(x)=x^2$", tex_template=ctex_template, color=BLUE),
            Tex(r"$g(x)=x^3$", tex_template=ctex_template, color=RED),
            Tex(r"$h(x)=x^5$", tex_template=ctex_template, color=PURPLE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05).scale(0.4)
        legend.next_to(axes, UP, buff=0.1).align_to(axes, RIGHT)
        
        # 显示坐标系和函数图像
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(f_graph), Write(legend[0]))
        self.play(Create(g_graph), Write(legend[1]))
        self.play(Create(h_graph), Write(legend[2]))
        self.wait(1)
        
        # 乘积法则公式 - 减小字体和间距
        product_rule = VGroup(
            Tex(r"\text{步骤2: 应用乘积法则}", tex_template=ctex_template, color=YELLOW),
            Tex(r"$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$", tex_template=ctex_template)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).scale(0.6)
        product_rule.next_to(function_def, DOWN, buff=0.25)
        
        # 显示乘积法则
        self.play(Write(product_rule[0]))
        self.play(Write(product_rule[1]))
        self.wait(1)
        
        # 计算各个导数 - 减小字体和间距
        derivatives = VGroup(
            Tex(r"\text{步骤3: 计算各部分导数}", tex_template=ctex_template, color=YELLOW),
            Tex(r"$f'(x) = \frac{d}{dx}(x^2) = 2x$", tex_template=ctex_template, color=BLUE),
            Tex(r"$g'(x) = \frac{d}{dx}(x^3) = 3x^2$", tex_template=ctex_template, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).scale(0.6)
        derivatives.next_to(product_rule, DOWN, buff=0.25)
        
        # 显示导数计算
        self.play(Write(derivatives[0]))
        self.play(Write(derivatives[1]))
        self.play(Write(derivatives[2]))
        self.wait(1)
        
        # 代入乘积法则公式 - 减小字体和间距，水平排列部分内容
        substitution_part1 = VGroup(
            Tex(r"\text{步骤4: 代入乘积法则公式}", tex_template=ctex_template, color=YELLOW),
            Tex(r"$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$", tex_template=ctex_template)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).scale(0.6)
        
        substitution_part2 = VGroup(
            Tex(r"$= 2x \cdot x^3 + x^2 \cdot 3x^2$", tex_template=ctex_template),
            Tex(r"$= 2x^4 + 3x^4 = 5x^4$", tex_template=ctex_template, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).scale(0.6)
        
        # 水平排列两部分
        substitution = VGroup(substitution_part1, substitution_part2).arrange(RIGHT, buff=0.5, aligned_edge=UP)
        substitution.next_to(derivatives, DOWN, buff=0.25)
        
        # 显示代入过程 - 简化动画步骤
        self.play(Write(substitution_part1[0]))
        self.play(Write(substitution_part1[1]))
        self.play(Write(substitution_part2[0]))
        self.play(Write(substitution_part2[1]))
        self.wait(1)
        
        # 显示最终导数图像 - 减小图例大小
        derivative_graph = axes.plot(lambda x: 5*x**4, color=GREEN)
        derivative_legend = Tex(r"$h'(x)=5x^4$", tex_template=ctex_template, color=GREEN).scale(0.4)
        derivative_legend.next_to(legend, DOWN, buff=0.05).align_to(legend, LEFT)
        
        self.play(Create(derivative_graph), Write(derivative_legend))
        self.wait(1)
        
        # 使用SymPy验证结果和结论 - 水平排列以节省垂直空间
        # 创建SymPy验证部分
        sympy_box = Rectangle(height=2, width=5.5, color=YELLOW, fill_opacity=0.1)
        sympy_box.to_edge(DOWN, buff=0.2).shift(LEFT*3)
        
        sympy_verification = VGroup(
            Tex(r"\text{使用SymPy验证:}", tex_template=ctex_template, color=YELLOW),
            Tex(r"$x = Symbol('x')$", tex_template=ctex_template),
            Tex(r"$y = x^2 \cdot x^3 = x^5$", tex_template=ctex_template),
            Tex(r"$y' = \frac{d}{dx}(y) = 5x^4$", tex_template=ctex_template, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).scale(0.5)
        sympy_verification.move_to(sympy_box.get_center())
        
        # 创建结论部分
        conclusion_box = Rectangle(height=2, width=5.5, color=GREEN, fill_opacity=0.1)
        conclusion_box.to_edge(DOWN, buff=0.2).shift(RIGHT*3)
        
        conclusion = VGroup(
            Tex(r"\text{结论:}", tex_template=ctex_template, color=GREEN),
            Tex(r"$\frac{d}{dx}(x^2 \cdot x^3) = 5x^4$", tex_template=ctex_template)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).scale(0.6)
        conclusion.move_to(conclusion_box.get_center())
        
        # 同时显示验证和结论
        self.play(Create(sympy_box), Create(conclusion_box))
        
        # 显示SymPy验证内容
        self.play(Write(sympy_verification))
        
        # 显示结论内容
        self.play(Write(conclusion))
        self.wait(2)

# 运行命令:
# python -m manim -pql product_rule_visual.py ProductRuleVisual