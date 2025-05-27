import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体和解决符号问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_fedavg_improved():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # ========== 1. 中央服务器 ==========
    server = patches.Rectangle((4, 2), 2, 2, linewidth=1.5, 
                              edgecolor='#2c3e50', facecolor='#3498db', alpha=0.9)
    ax.add_patch(server)
    plt.text(5, 3, '中央服务器\n全局模型: $w_t$', 
             ha='center', va='center', fontsize=11, color='white', weight='bold')
    
    # ========== 2. 客户端 ==========
    client_colors = ['#2ecc71', '#27ae60', '#16a085', '#1abc9c']  # 不同绿色系
    clients = []
    for i, (x, y) in enumerate([(1, 1), (1, 4), (7, 1), (7, 4)]):
        client = patches.Rectangle((x, y), 2, 1.5, linewidth=1, 
                                 edgecolor='#2c3e50', facecolor=client_colors[i], alpha=0.8)
        ax.add_patch(client)
        plt.text(x+1, y+0.75, f'客户端 {i+1}\n本地数据: $n_{i+1}$', 
                 ha='center', va='center', fontsize=9, color='white')
        clients.append((x+1, y+1.5))
    
    # ========== 3. 通信箭头 ==========
    arrow_style = dict(arrowstyle='->', linewidth=2, 
                       connectionstyle='arc3,rad=0.1', shrinkA=5, shrinkB=5)
    
    # 下发模型（红色实线箭头）
    for (cx, cy) in clients:
        ax.annotate("", xy=(cx, cy-0.2), xytext=(5, 2.5),
                    arrowprops=dict(color='#e74c3c', **arrow_style))
    
    # 上传更新（蓝色实线箭头）
    for (cx, cy) in clients:
        ax.annotate("", xy=(5, 3.5), xytext=(cx, cy-0.5),
                    arrowprops=dict(color='#2980b9', **arrow_style))
    
    # ========== 4. 图例说明 ==========
    plt.plot([], [], color='#e74c3c', linewidth=2, label='下发全局模型')
    plt.plot([], [], color='#2980b9', linewidth=2, label='上传本地更新')
    plt.legend(loc='upper right', framealpha=0.9)
    
    # ========== 5. 聚合公式 ==========
    plt.text(5, 0.7, r'模型聚合: $w^{t+1}_{global} = \sum_{k=1}^C \frac{|n_k|}{|N|} w_{global}^{t+1}$', 
             ha='center', va='center', fontsize=11, 
             bbox=dict(facecolor='#f39c12', alpha=0.7, edgecolor='none'))
    
    plt.title('联邦平均算法 (FedAvg) 框架', pad=20, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('fedavg_improved.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

draw_fedavg_improved()