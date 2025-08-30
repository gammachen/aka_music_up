from sympy import *
import matplotlib.pyplot as plt
import numpy as np

def visualize_product_rule():
    # 定义符号变量
    x = Symbol('x')
    
    # 定义函数
    print("步骤1: 定义函数")
    f = x**2
    g = x**3
    h = f * g
    
    print(f"f(x) = {f}")
    print(f"g(x) = {g}")
    print(f"h(x) = f(x) * g(x) = {h}")
    print()
    
    # 计算各个导数
    print("步骤2: 计算各部分导数")
    f_prime = diff(f, x)
    g_prime = diff(g, x)
    
    print(f"f'(x) = {f_prime}")
    print(f"g'(x) = {g_prime}")
    print()
    
    # 应用乘积法则
    print("步骤3: 应用乘积法则 (f*g)' = f'*g + f*g'")
    product_rule_result = f_prime * g + f * g_prime
    print(f"(f*g)' = f'*g + f*g'")
    print(f"      = {f_prime} * {g} + {f} * {g_prime}")
    print(f"      = {f_prime * g} + {f * g_prime}")
    print(f"      = {product_rule_result}")
    print()
    
    # 直接计算导数验证
    print("步骤4: 直接计算h'(x)验证结果")
    h_prime = diff(h, x)
    print(f"h'(x) = {h_prime}")
    print()
    
    # 可视化函数和导数
    print("步骤5: 绘制函数图像")
    x_vals = np.linspace(0, 2, 100)
    f_vals = [float(f.subs(x, val)) for val in x_vals]
    g_vals = [float(g.subs(x, val)) for val in x_vals]
    h_vals = [float(h.subs(x, val)) for val in x_vals]
    h_prime_vals = [float(h_prime.subs(x, val)) for val in x_vals]
    
    plt.figure(figsize=(12, 8))
    
    # 原函数图像
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, f_vals, 'b-', label=f'f(x) = {f}')
    plt.plot(x_vals, g_vals, 'r-', label=f'g(x) = {g}')
    plt.plot(x_vals, h_vals, 'g-', label=f'h(x) = f(x)*g(x) = {h}')
    plt.grid(True)
    plt.legend()
    plt.title('函数图像')
    
    # 导数图像
    plt.subplot(2, 1, 2)
    plt.plot(x_vals, h_prime_vals, 'm-', label=f"h'(x) = {h_prime}")
    plt.grid(True)
    plt.legend()
    plt.title('导数图像')
    
    plt.tight_layout()
    plt.savefig('/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/manim/product_rule_result.png')
    print("图像已保存为 product_rule_result.png")
    
    # 可选：显示图像
    plt.show()

if __name__ == "__main__":
    print("乘积求导法则可视化")
    print("计算 d[f(x)g(x)]/dx，其中 f(x)=x^2, g(x)=x^3")
    print("-" * 50)
    visualize_product_rule()