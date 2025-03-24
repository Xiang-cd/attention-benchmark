import streamlit as st
import pandas as pd
import json
import argparse

def load_benchmark_data(file):
    # 读取 JSON 文件
    with open(file, 'r') as f:
        data = json.load(f)
    data = data['res_ls']
    # 将 JSON 数据转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    
    # 确保数据结构与示例数据一致
    required_columns = ['seq_len', 'batch_size', 'headdim', 'numhead', 'algorithm', 'time', 'flops', 'mode']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='json file')
    args = parser.parse_args()
    df = load_benchmark_data(args.file)
    with open(args.file, 'r') as f:
        data = json.load(f)
    device_name = data['device_name']
    st.title(f"Attention Benchmark Visualization {args.file} {device_name}")
    
    # 加载数据
    
    # 侧边栏控制
    st.sidebar.header("参数设置")
    metric = st.sidebar.selectbox(
        "选择指标",
        ["time", "flops"]
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
    
    selected_numhead = st.sidebar.multiselect(
        "选择Head Dimension",
        options=df['numhead'].unique(),
        default=df['numhead'].unique()
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
        (df['batch_size'].isin(selected_batch_size)) &
        (df['numhead'].isin(selected_numhead))
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
        index='seq_len'
    )
    
    st.subheader(f"Sequence Length vs {metric}")
    st.line_chart(pivot_df)
        

if __name__ == "__main__":
    main()