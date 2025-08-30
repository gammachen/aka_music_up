from manim import *
from manim.utils.tex_templates import TexTemplate

class EquationSolverVisual(Scene):
    def construct(self):
        # 中文公式模板配置
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble='\\usepackage{amsmath}\\usepackage{ctex}\\usepackage{fontspec}\\setmainfont{STHeiti}\\setromanfont{STHeiti}\\setsansfont{STHeiti}\\setmonofont{Menlo}\\usepackage{xeCJK}\\setCJKmainfont{STHeiti}\\setCJKsansfont{STHeiti}\\setCJKmonofont{STHeiti}'
        )

        # 创建标题
        title = Tex(r"\text{方程求解演示：3x + x/2 = 28}", tex_template=ctex_template, font_size=40).to_edge(UP)

        # 初始化图形元素
        square = Square(side_length=0.8, color=BLUE, fill_opacity=0.8)  # 代表2个圆
        half_square = Rectangle(height=0.8, width=0.4, color=GREEN, fill_opacity=0.8)
        circle = Circle(radius=0.3, color=RED, fill_opacity=0.8)

        # 创建初始方程可视化
        term_3x = VGroup(*[square.copy() for _ in range(3)]).arrange(RIGHT, buff=0.2)
        term_x2 = VGroup(half_square.copy()).next_to(term_3x, RIGHT, buff=0.5)
        equation = VGroup(
            term_3x,
            Tex("+", tex_template=ctex_template),
            term_x2,
            Tex("=", tex_template=ctex_template),
            Tex("28", tex_template=ctex_template, font_size=36)
        ).arrange(RIGHT, buff=0.3).scale(0.9).shift(UP*0.5)

        # 分步说明文本 - 放在右侧以避免与动画元素重叠
        steps = VGroup(
            Tex(r"1. 每个方块代表2个圆", tex_template=ctex_template, color=YELLOW),
            Tex(r"2. 半块转换为1个圆", tex_template=ctex_template, color=YELLOW),
            Tex(r"3. 总共7个圆 = 28", tex_template=ctex_template, color=YELLOW),
            Tex(r"4. 每个圆 = 4 → 方块 = 8", tex_template=ctex_template, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).scale(0.7).to_edge(RIGHT).shift(UP*1)

        # 动画流程
        self.play(Write(title))
        self.play(LaggedStart(
            Create(term_3x),
            Create(term_x2),
            Write(equation[1]),
            Write(equation[3]),
            Write(equation[4]),
            lag_ratio=0.4
        ))
        self.wait(2)

        # 方块转圆动画 - 增加间距并调整布局避免重叠
        circle_group = VGroup()
        # 创建一个专门的转换区域，位于屏幕中央偏下位置
        conversion_area = VGroup().move_to(DOWN*1.5)
        
        # 创建一个说明标签，与转换区域保持距离
        conversion_label = Tex(r"方块转换为圆形", tex_template=ctex_template, color=WHITE)
        conversion_label.next_to(conversion_area, UP, buff=0.5)
        self.play(Write(conversion_label))
        
        # 为每个方块创建转换动画，确保元素间有足够间距
        for i, square in enumerate(term_3x):
            # 计算每个转换位置，确保水平分布且不重叠
            position = conversion_area.get_center() + RIGHT*(i-1)*2.5
            
            # 创建两个圆并排列
            circles = VGroup(*[circle.copy() for _ in range(2)]).arrange(RIGHT, buff=0.4)
            circles.move_to(position)
            
            # 添加到圆形组
            circle_group.add(circles)
            
            # 播放转换动画
            self.play(
                square.animate.move_to(position + LEFT*0.8),  # 方块移到圆的左侧
                ReplacementTransform(square.copy(), circles),
                FadeIn(Tex(r"=", tex_template=ctex_template).move_to(position + LEFT*0.4))
            )
            
            # 高亮显示当前步骤说明
            step_highlight = steps[0].copy().set_color(RED)
            self.play(FadeIn(step_highlight))
            self.play(FadeOut(step_highlight))
        
        # 半块转圆动画 - 清晰展示转换过程
        # 创建一个新的位置，确保与前面的转换区域不重叠
        half_block_position = conversion_area.get_center() + RIGHT*4
        
        # 移动半块到指定位置
        self.play(term_x2.animate.move_to(half_block_position + LEFT*0.5))
        
        # 创建转换后的圆
        half_to_circle = circle.copy().move_to(half_block_position + RIGHT*0.5)
        
        # 添加等号连接半块和圆
        half_equal = Tex(r"=", tex_template=ctex_template).move_to(half_block_position)
        
        # 播放转换动画
        self.play(
            Write(half_equal),
            FadeIn(half_to_circle)
        )
        self.play(
            ReplacementTransform(term_x2, half_to_circle),
            FadeOut(half_equal)
        )
        
        # 高亮显示当前步骤说明
        step_highlight = steps[1].copy().set_color(RED)
        self.play(FadeIn(step_highlight))
        self.play(FadeOut(step_highlight))
        
        # 添加到圆形组
        circle_group.add(half_to_circle)
        
        # 合并圆形等式 - 整齐排列所有圆形
        # 清除转换标签
        self.play(FadeOut(conversion_label))
        
        # 将所有圆形整齐排列
        final_circles = VGroup(*circle_group).arrange(RIGHT, buff=0.3).scale(0.8)
        
        # 创建完整等式，确保与其他元素有足够间距
        equation_circles = VGroup(
            final_circles,
            Tex("=", tex_template=ctex_template),
            Tex("28", tex_template=ctex_template, font_size=36)
        ).arrange(RIGHT, buff=0.5).move_to(DOWN*0.5)
        
        # 添加说明标签
        equation_label = Tex(r"圆形等式", tex_template=ctex_template, color=WHITE)
        equation_label.next_to(equation_circles, UP, buff=0.5)
        
        # 播放合并动画
        self.play(
            Write(equation_label),
            Transform(circle_group, equation_circles)
        )
        
        # 高亮显示当前步骤说明
        step_highlight = steps[2].copy().set_color(RED)
        self.play(FadeIn(step_highlight))
        self.play(FadeOut(step_highlight))
        
        self.wait()

        # 数值分解动画 - 实现完整的数值分解过程
        # 清除前面的标签
        self.play(FadeOut(equation_label))
        
        # 创建数值分解区域，确保与其他元素不重叠
        value_area = DOWN*2
        
        # 创建数值分解标签
        value_label = Tex(r"计算每个圆的值", tex_template=ctex_template, color=WHITE)
        value_label.next_to(value_area, UP, buff=0.5)
        self.play(Write(value_label))
        
        # 创建7个圆形表示等式左边
        value_circles = VGroup(*[circle.copy() for _ in range(7)])
        value_circles.arrange(RIGHT, buff=0.2).scale(0.7).move_to(value_area + LEFT*2)
        
        # 创建28的数值表示
        value_number = Tex("28", tex_template=ctex_template, font_size=36)
        value_number.move_to(value_area + RIGHT*2)
        
        # 创建等号
        value_equal = Tex("=", tex_template=ctex_template)
        value_equal.move_to(value_area)
        
        # 播放数值分解动画
        self.play(
            FadeIn(value_circles),
            Write(value_equal),
            FadeIn(value_number)
        )
        
        # 为每个圆添加数值4
        for i, circ in enumerate(value_circles):
            value_text = Tex("4", tex_template=ctex_template, font_size=24, color=WHITE)
            value_text.move_to(circ)
            self.play(Write(value_text), run_time=0.3)
        
        # 高亮显示最后一步说明
        step_highlight = steps[3].copy().set_color(RED)
        self.play(FadeIn(step_highlight))
        self.play(FadeOut(step_highlight))
        
        # 创建最终结论
        conclusion = Tex(r"解得：x = 8", tex_template=ctex_template, color=GREEN, font_size=48)
        conclusion.to_edge(DOWN, buff=0.5)
        self.play(Write(conclusion))
        
        self.wait(3)

# 运行命令：manim -pql equation_solver_visual.py EquationSolverVisual