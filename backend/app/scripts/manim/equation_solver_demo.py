from manim import *
from manim.utils.tex_templates import TexTemplate

class EquationSolver(Scene):
    def construct(self):
        # 配置中文公式模板
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble='\\usepackage{amsmath}\\usepackage{ctex}\\usepackage{fontspec}\\setmainfont{STHeiti}\\setromanfont{STHeiti}\\setsansfont{STHeiti}\\setmonofont{Menlo}\\usepackage{xeCJK}\\setCJKmainfont{STHeiti}\\setCJKsansfont{STHeiti}\\setCJKmonofont{STHeiti}'
        )

        # 创建标题
        title = Tex(r"\text{方程求解演示：3x + x/2 = 28}", tex_template=ctex_template, font_size=40).to_edge(UP)
        
        # 初始化方块组
        x_block = Square(side_length=0.6, color=BLUE, fill_opacity=0.8)
        half_block = Rectangle(height=0.6, width=0.3, color=GREEN, fill_opacity=0.8)

        # 创建代数项可视化
        term_3x = VGroup(*[x_block.copy() for _ in range(3)]).arrange(RIGHT, buff=0.1)
        term_x2 = VGroup(half_block.copy()).next_to(term_3x, RIGHT, buff=0.5)
        
        # 创建方程可视化
        equation = VGroup(
            term_3x,
            Tex("+", tex_template=ctex_template),
            term_x2,
            Tex("=", tex_template=ctex_template),
            Tex("28", tex_template=ctex_template)
        ).arrange(RIGHT, buff=0.3).scale(0.8).shift(UP*0.5)

        # 创建分步说明文本
        steps = VGroup(
            Tex(r"1. 合并同类项：3x + 0.5x = 3.5x", tex_template=ctex_template, color=YELLOW),
            Tex(r"2. 两边同时除以3.5：x = 28 ÷ 3.5", tex_template=ctex_template, color=YELLOW),
            Tex(r"3. 计算结果：x = 8", tex_template=ctex_template, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).scale(0.7).to_edge(LEFT)

        # 动画流程
        self.play(Write(title))
        self.play(LaggedStart(
            Create(term_3x),
            Create(term_x2),
            Write(equation[1]),
            Write(equation[3]),
            Write(equation[4]),
            lag_ratio=0.3
        ))
        self.wait(2)

        # 合并同类项动画
        combined_blocks = VGroup(
            *[x_block.copy() for _ in range(3)],
            half_block.copy()
        ).arrange(RIGHT, buff=0.1)
        combined_text = Tex("3.5x", tex_template=ctex_template, color=PURPLE).next_to(combined_blocks, DOWN)
        self.play(
            Transform(term_3x.copy(), combined_blocks),
            Transform(term_x2.copy(), combined_blocks[3]),
            Write(combined_text),
            Write(steps[0])
        )
        self.wait()

        # 数字分解动画
        number_blocks = VGroup(*[
            Square(side_length=0.6, color=ORANGE, fill_opacity=0.6)
            for _ in range(8)
        ]).arrange_in_grid(rows=2, cols=4, buff=0.2).shift(DOWN*1.5)
        
        self.play(
            equation[4].animate.move_to(number_blocks.get_center()),
            FadeIn(number_blocks),
            Write(steps[1])
        )
        self.wait()

        # 最终解动画
        solution_text = Tex(r"x = 8", tex_template=ctex_template, color=RED, font_size=48)
        arrow = Arrow(steps[2].get_bottom(), solution_text.get_top(), color=YELLOW)
        
        self.play(
            number_blocks.animate.arrange(RIGHT, buff=0.1).scale(0.7),
            Write(steps[2]),
            GrowArrow(arrow),
            Write(solution_text.next_to(arrow, RIGHT))
        )
        self.wait(3)

# 运行命令：manim -pql equation_solver_demo.py EquationSolver