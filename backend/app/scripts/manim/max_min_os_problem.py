from manim import *

class SimplexMethodDemo(Scene):
    def construct(self):
        # 初始化坐标轴
        axes = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 90, 10],
            x_length=6,
            y_length=6,
            axis_config={"stroke_width": 2},
            tips=False
        )
        self.add(axes)
        
        # 添加标题
        title = Text("单纯形法演示").scale(0.8).to_edge(UP)
        self.add(title)

        # 绘制约束线
        constraints = [
            {"func": lambda x: 100 - 2*x, "color": BLUE, "label": "2x + y = 100"},
            {"func": lambda x: 80 - x, "color": GREEN, "label": "x + y = 80"},
            {"x": 40, "color": RED, "label": "x = 40"}
        ]

        lines = []
        for constraint in constraints:
            if "func" in constraint:
                line = axes.plot(constraint["func"], color=constraint["color"])
                label = axes.get_graph_label(line, constraint["label"])
            else:
                line = Line(
                    axes.coords_to_point(constraint["x"], 0),
                    axes.coords_to_point(constraint["x"], 80),
                    color=constraint["color"]
                )
                label = Text(constraint["label"]).next_to(line, UP)
            lines.append((line, label))
        
        for line, label in lines:
            self.add(line, label)

        # 填充可行区域
        feasible_points = [
            axes.coords_to_point(20, 60),
            axes.coords_to_point(40, 20),
            axes.coords_to_point(40, 0),
            axes.coords_to_point(0, 80)
        ]
        feasible_region = Polygon(*feasible_points, color=YELLOW)
        feasible_region.set_fill(YELLOW, opacity=0.5)
        self.add(feasible_region)

        # 创建动态目标函数
        self.c = 0  # 初始z值
        def z_func(x):
            return (self.c - 3*x)/2
        
        z_line = always_redraw(lambda: axes.plot(
            z_func,
            x_range=[0, 50],
            color=PURPLE
        ))
        self.add(z_line)

        # 移动目标函数等值线
        def update_z_line(z_line, new_c):
            self.c = new_c
            return z_line

        # 创建一个函数，根据alpha值更新目标函数
        def get_z_func_with_alpha(alpha):
            return lambda x: (self.c + 50*alpha - 3*x)/2

        self.play(
            UpdateFromAlphaFunc(
                z_line,
                lambda mob, alpha: mob.become(
                    axes.plot(get_z_func_with_alpha(alpha), x_range=[0, 50], color=PURPLE)
                )
            ),
            run_time=3
        )
        self.c += 50

        # 添加最优解标记
        optimal_point = Dot(axes.coords_to_point(20, 60), color=RED)
        optimal_label = Text("(20, 60)", color=RED).next_to(optimal_point, DOWN)
        x_line = Line(optimal_point, axes.coords_to_point(20, 0), color=RED, stroke_width=1)
        y_line = Line(optimal_point, axes.coords_to_point(0, 60), color=RED, stroke_width=1)
        
        self.add(optimal_point, optimal_label, x_line, y_line)

        # 创建单纯形法表格
        table_data = [
            ["基变量", "x₁", "x₂", "s₁", "s₂", "s₃", "RHS"],
            ["s₁", "2", "1", "1", "0", "0", "100"],
            ["s₂", "1", "1", "0", "1", "0", "80"],
            ["s₃", "1", "0", "0", "0", "1", "40"],
            ["-z", "-3", "-2", "0", "0", "0", "0"]
        ]
        
        table = Table(table_data).scale(0.6).to_edge(RIGHT)
        table.add_background_rectangle()
        self.add(table)

        # 添加箭头指示
        arrow = Arrow(table.get_center(), optimal_point, color=WHITE)
        arrow_label = Text("最优解").next_to(arrow, DOWN)
        self.add(arrow, arrow_label)

        self.wait(3)

#  运行命令：manim -pql max_min_os_problem.py SimplexMethodDemo