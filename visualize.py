import streamlit as st
import pandas as pd
import plotly.express as px

def load_benchmark_data():
    # 示例数据结构
    data = {
        'seqlen': [128, 256, 512] * 2,  # 重复以覆盖不同组合
        'batch_size': [1, 1, 1, 2, 2, 2],
        'headdim': [32, 32, 32] * 2,
        'algorithm': ['flash', 'flash', 'flash', 'vanilla', 'vanilla', 'vanilla'],
        'time_ms': [1, 10, 20, 2, 12, 24],  # 示例时间数据
        'flops': [10, 15, 20, 12, 18, 24],  # 示例FLOPS数据
    }
    return pd.DataFrame(data)

def main():
    st.title("Attention Benchmark Visualization")
    
    # 加载数据
    df = load_benchmark_data()
    
    # 侧边栏控制
    st.sidebar.header("参数设置")
    metric = st.sidebar.selectbox(
        "选择指标",
        ["time_ms", "flops"]
    )
    
    # 添加算法选择
    selected_algorithms = st.sidebar.multiselect(
        "选择算法",
        options=df['algorithm'].unique(),
        default=df['algorithm'].unique()
    )
    
    # 添加headdim选择
    selected_headdims = st.sidebar.multiselect(
        "选择Head Dimension",
        options=df['headdim'].unique(),
        default=df['headdim'].unique()
    )
    
    selected_batch_size = st.sidebar.multiselect(
        "选择batch size",
        options=df['batch_size'].unique(),
        default=df['batch_size'].unique()
    )
    
    
    # 过滤数据
    filtered_df = df[
        (df['algorithm'].isin(selected_algorithms)) &
        (df['headdim'].isin(selected_headdims)) & 
        (df['batch_size'].isin(selected_batch_size))
    ]
    
        # 为每个算法和headdim组合创建一个唯一标识
    filtered_df['group'] = filtered_df.apply(
        lambda x: f"{x['algorithm']}_headdim_{x['headdim']}", 
        axis=1
    )
    
    # 创建透视表
    pivot_df = filtered_df.pivot(
        columns='group',
        values=metric,
        index='seqlen'
    )
    
    st.subheader(f"Sequence Length vs {metric}")
    st.line_chart(pivot_df)
        

if __name__ == "__main__":
    main()