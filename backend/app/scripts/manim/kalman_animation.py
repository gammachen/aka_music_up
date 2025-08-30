from manim import *
from manim.utils.tex_templates import TexTemplate

class KalmanFilterAnimation(Scene):
    def construct(self):
        # 定义所有公式
        ctex_template = TexTemplate(
            tex_compiler='xelatex',
            output_format='.xdv',
            preamble='\\usepackage{amsmath}\\usepackage{ctex}\\usepackage{fontspec}\\setmainfont{STHeiti}\\setromanfont{STHeiti}\\setsansfont{STHeiti}\\setmonofont{Menlo}\\usepackage{xeCJK}\\setCJKmainfont{STHeiti}\\setCJKsansfont{STHeiti}\\setCJKmonofont{STHeiti}'
        )
        title = Tex(r"\text{Kalman滤波器}", tex_template=ctex_template, font_size=40).to_edge(UP)
        
        equations = VGroup(
            Tex(r"1. 状态预测: $X_{\text{prior}} = A \cdot X_{\text{posterior}}^{(k-1)}$", tex_template=ctex_template),
            Tex(r"2. 协方差预测: $P_{\text{prior}} = A P_{\text{posterior}}^{(k-1)} A^T + Q$", tex_template=ctex_template),
            Tex(r"3. 卡尔曼增益: $K = P_{\text{prior}} H^T (H P_{\text{prior}} H^T + R)^{-1}$", tex_template=ctex_template),
            Tex(r"4. 状态更新: $X_{\text{posterior}}^{(k)} = X_{\text{prior}} + K(Z - H X_{\text{prior}})$", tex_template=ctex_template),
            Tex(r"5. 协方差更新: $P_{\text{posterior}} = (I - K H) P_{\text{prior}}$", tex_template=ctex_template),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).scale(0.8).to_edge(LEFT)

        # 创建可视化元素
        state_circle = Circle(radius=0.5, color=BLUE).shift(RIGHT*3 + UP)
        cov_ellipse = Ellipse(width=1.5, height=1.0, color=RED).move_to(state_circle)
        measurement = Dot(color=YELLOW).shift(RIGHT*3 + DOWN)
        kalman_gain_arrow = Arrow(ORIGIN, RIGHT, color=GREEN)

        # 初始状态
        self.play(Write(title))
        self.play(Create(state_circle), Create(cov_ellipse))
        self.wait(1)

        # 逐步展示方程
        for i, eq in enumerate(equations):
            self.play(Write(eq))
            self.animate_step(i+1, state_circle, cov_ellipse, measurement, kalman_gain_arrow)
            self.wait(1)

    def animate_step(self, step, state, cov, meas, kg_arrow):
        # 各步骤动画效果
        if step == 1:
            self.play(
                state.animate.shift(UR*0.5),
                cov.animate.stretch(1.2, 0).shift(UR*0.5),
                run_time=2
            )
        elif step == 2:
            self.play(
                cov.animate.stretch(1.5, 0).stretch(1.2, 1),
                run_time=2
            )
        elif step == 3:
            kg_arrow.next_to(state, DOWN)
            self.play(
                Create(meas),
                GrowArrow(kg_arrow),
                run_time=2
            )
        elif step == 4:
            self.play(
                state.animate.move_to(interpolate(
                    state.get_center(),
                    meas.get_center(),
                    0.3
                )),
                cov.animate.scale(0.8),
                run_time=2
            )
        elif step == 5:
            self.play(
                cov.animate.scale(0.7).rotate(0.2),
                FadeOut(meas),
                run_time=2
            )

# 运行命令（在命令行中使用）:
# manim -pql kalman_animation.py KalmanFilterAnimation