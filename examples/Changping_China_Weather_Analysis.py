#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Polars 分析北京昌平区空气质量数据
=====================================

本脚本演示了如何使用 Polars 对北京昌平区的空气质量数据进行分析。
数据包含 PM2.5、PM10、SO2、NO2、CO、O3 等污染物浓度以及气象数据。

数据字段说明：
- No: 序号
- year: 年份
- month: 月份
- day: 日期
- hour: 小时
- PM2.5: PM2.5 浓度 (μg/m³)
- PM10: PM10 浓度 (μg/m³)
- SO2: 二氧化硫浓度 (μg/m³)
- NO2: 二氧化氮浓度 (μg/m³)
- CO: 一氧化碳浓度 (μg/m³)
- O3: 臭氧浓度 (μg/m³)
- TEMP: 温度 (°C)
- PRES: 气压 (hPa)
- DEWP: 露点温度 (°C)
- RAIN: 降水量 (mm)
- wd: 风向
- WSPM: 风速 (m/s)
- station: 监测站名称

作者: [您的姓名]
创建日期: 2025年
"""

import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """
    加载空气质量数据
    
    参数:
        file_path (str): CSV 文件路径
        
    返回:
        DataFrame: Polars DataFrame 格式的数据
    """
    try:
        # 定义数据类型以避免解析错误
        dtypes = {
        "No": pl.Int64,
        "year": pl.Int64,
        "month": pl.Int64,
        "day": pl.Int64,
        "hour": pl.Int64,
        "PM2.5": pl.Float64,
        "PM10": pl.Float64,
        "SO2": pl.Float64,
        "NO2": pl.Float64,
        "CO": pl.Float64,
        "O3": pl.Float64,
        "TEMP": pl.Float64,
        "PRES": pl.Float64,
        "DEWP": pl.Float64,
        "RAIN": pl.Float64,
        "wd": pl.Utf8,  # 使用 Utf8 而不是 String
        "WSPM": pl.Float64,
        "station": pl.Utf8  # 使用 Utf8 而不是 String
    }
        
        # 使用 Polars 读取 CSV 数据，指定数据类型和缺失值处理
        df = pl.read_csv(file_path, schema_overrides=dtypes, null_values=["NA", "NaN", "null", ""])
        print(f"成功加载数据，共有 {df.height} 行，{df.width} 列")
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def explore_data(df):
    """
    探索数据的基本信息
    
    参数:
        df (DataFrame): Polars DataFrame
    """
    print("\n=== 数据基本信息 ===")
    print(f"数据形状: ({df.height}, {df.width})")
    
    print("\n=== 前5行数据 ===")
    print(df.head())
    
    print("\n=== 数据类型 ===")
    print(df.dtypes)
    
    print("\n=== 缺失值统计 ===")
    null_counts = df.null_count()
    print(null_counts)
    
    print("\n=== 数值列的基本统计信息 ===")
    # 获取数值类型的列
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                               pl.Float32, pl.Float64)]
    if numeric_cols:
        print(df.select(numeric_cols).describe())
    else:
        print("未找到数值类型的列")

def preprocess_data(df):
    """
    数据预处理
    
    参数:
        df (DataFrame): 原始数据
        
    返回:
        DataFrame: 处理后的数据
    """
    # 创建日期时间列
    df = df.with_columns(
        pl.datetime(pl.col("year"), pl.col("month"), pl.col("day"), pl.col("hour")).alias("datetime")
    )
    
    # 按日期排序
    df = df.sort("datetime")
    
    # 处理缺失值 - 使用前向填充填补缺失值
    df = df.fill_null(strategy="forward")
    
    print("数据预处理完成")
    return df

def analyze_pollutants(df):
    """
    分析污染物浓度
    
    参数:
        df (DataFrame): 处理后的数据
    """
    print("\n=== 污染物浓度分析 ===")
    
    # 计算各污染物的平均值
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    avg_pollutants = df.select([
        pl.col(pollutant).mean().alias(f"{pollutant}_mean") 
        for pollutant in pollutants
    ])
    
    print("各污染物平均浓度:")
    print(avg_pollutants)
    
    # 计算空气质量指数级别
    # 根据中国标准计算 AQI
    def calculate_aqi_pm25(pm25):
        """计算 PM2.5 对应的 AQI"""
        if pm25 <= 0:
            return 0
        elif pm25 <= 35:
            return pm25 * 50 / 35
        elif pm25 <= 75:
            return (pm25 - 35) * 50 / 40 + 50
        elif pm25 <= 115:
            return (pm25 - 75) * 50 / 40 + 100
        elif pm25 <= 150:
            return (pm25 - 115) * 50 / 35 + 150
        elif pm25 <= 250:
            return (pm25 - 150) * 100 / 100 + 200
        else:
            return (pm25 - 250) * 100 / 100 + 300
    
    # 添加 AQI 列（简化计算）
    df = df.with_columns([
        pl.when(pl.col("PM2.5").is_null())
        .then(None)
        .otherwise(
            pl.col("PM2.5").map_elements(calculate_aqi_pm25, return_dtype=pl.Float64)
        ).alias("AQI_PM25")
    ])
    
    # 统计 AQI 级别分布
    aqi_stats = df.select([
        pl.col("AQI_PM25").filter(pl.col("AQI_PM25") <= 50).count().alias("优"),
        pl.col("AQI_PM25").filter((pl.col("AQI_PM25") > 50) & (pl.col("AQI_PM25") <= 100)).count().alias("良"),
        pl.col("AQI_PM25").filter((pl.col("AQI_PM25") > 100) & (pl.col("AQI_PM25") <= 150)).count().alias("轻度污染"),
        pl.col("AQI_PM25").filter((pl.col("AQI_PM25") > 150) & (pl.col("AQI_PM25") <= 200)).count().alias("中度污染"),
        pl.col("AQI_PM25").filter((pl.col("AQI_PM25") > 200) & (pl.col("AQI_PM25") <= 300)).count().alias("重度污染"),
        pl.col("AQI_PM25").filter(pl.col("AQI_PM25") > 300).count().alias("严重污染")
    ])
    
    print("\nAQI 级别分布:")
    print(aqi_stats)

def analyze_temporal_patterns(df):
    """
    分析时间模式
    
    参数:
        df (DataFrame): 处理后的数据
    """
    print("\n=== 时间模式分析 ===")
    
    # 按年份统计平均 PM2.5
    yearly_pm25 = df.group_by("year").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("year")
    
    print("年度平均 PM2.5 浓度:")
    print(yearly_pm25)
    
    # 按月份统计平均 PM2.5
    monthly_pm25 = df.group_by("month").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("month")
    
    print("\n月度平均 PM2.5 浓度:")
    print(monthly_pm25)
    
    # 按小时统计平均 PM2.5
    hourly_pm25 = df.group_by("hour").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("hour")
    
    print("\n小时平均 PM2.5 浓度:")
    print(hourly_pm25)

def visualize_data(df):
    """
    数据可视化
    
    参数:
        df (DataFrame): 处理后的数据
    """
    print("\n=== 生成可视化图表 ===")
    
    # 设置图形大小
    plt.figure(figsize=(15, 12))
    
    # 1. PM2.5 时间序列图
    plt.subplot(2, 3, 1)
    daily_pm25 = df.group_by(pl.col("datetime").dt.date()).agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("datetime")
    
    # 由于数据量大，我们只显示一部分
    sample_data = daily_pm25.sample(300).sort("datetime")
    plt.plot(sample_data["datetime"], sample_data["avg_PM2.5"], linewidth=0.8)
    plt.title("PM2.5 浓度时间序列（抽样）")
    plt.xlabel("日期")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.xticks(rotation=45)
    
    # 2. 各污染物平均浓度柱状图
    plt.subplot(2, 3, 2)
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    avg_values = [
        df.select(pl.col(pollutant).mean())[0, 0] 
        for pollutant in pollutants if pollutant in df.columns
    ]
    
    bars = plt.bar(pollutants, avg_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.title("各污染物平均浓度")
    plt.ylabel("浓度 (μg/m³)")
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, avg_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # 3. 年度趋势图
    plt.subplot(2, 3, 3)
    yearly_data = df.group_by("year").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("year")
    
    plt.plot(yearly_data["year"], yearly_data["avg_PM2.5"], marker='o', linewidth=2, markersize=6)
    plt.title("PM2.5 年度平均浓度趋势")
    plt.xlabel("年份")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True, alpha=0.3)
    
    # 4. 月度变化模式
    plt.subplot(2, 3, 4)
    monthly_data = df.group_by("month").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("month")
    
    plt.plot(monthly_data["month"], monthly_data["avg_PM2.5"], marker='s', linewidth=2, markersize=5)
    plt.title("PM2.5 月度变化模式")
    plt.xlabel("月份")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True, alpha=0.3)
    
    # 5. 小时变化模式
    plt.subplot(2, 3, 5)
    hourly_data = df.group_by("hour").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("hour")
    
    plt.plot(hourly_data["hour"], hourly_data["avg_PM2.5"], marker='d', linewidth=2, markersize=5, color='green')
    plt.title("PM2.5 小时变化模式")
    plt.xlabel("小时")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True, alpha=0.3)
    
    # 6. 温度与 PM2.5 的关系
    plt.subplot(2, 3, 6)
    # 抽样数据以提高性能
    sample_df = df.sample(1000).sort("TEMP")
    plt.scatter(sample_df["TEMP"], sample_df["PM2.5"], alpha=0.5, s=1)
    plt.title("温度与 PM2.5 浓度关系（抽样）")
    plt.xlabel("温度 (°C)")
    plt.ylabel("PM2.5 (μg/m³)")
    
    plt.tight_layout()
    plt.savefig("changping_air_quality_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 'changping_air_quality_analysis.png'")

def correlation_analysis(df):
    """
    相关性分析
    
    参数:
        df (DataFrame): 处理后的数据
    """
    print("\n=== 相关性分析 ===")
    
    # 选择数值列进行相关性分析
    numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    numeric_data = df.select(available_numeric_cols)
    
    # 计算相关系数矩阵
    if len(available_numeric_cols) > 1:
        # 使用 Polars 的 corr 方法计算相关系数矩阵
        corr_matrix = numeric_data.select([
            pl.corr(pl.col(col1), pl.col(col2)).alias(f"{col1}_vs_{col2}")
            for col1 in available_numeric_cols
            for col2 in available_numeric_cols
        ])
        print("相关系数矩阵:")
        print(corr_matrix)
    else:
        print("可用的数值列不足，无法计算相关系数矩阵")
    
    # 找出与 PM2.5 相关性最强的前5个变量
    # 这里我们采用更简单的方法直接计算与 PM2.5 的相关性
    if "PM2.5" in df.columns and len(available_numeric_cols) > 1:
        correlations = []
        for col in available_numeric_cols:
            if col != "PM2.5":
                # 计算与 PM2.5 的相关性
                corr_value = df.select(pl.corr(pl.col("PM2.5"), pl.col(col))).item()
                correlations.append((col, corr_value))
        
        # 按绝对值排序
        correlations.sort(key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
        
        print("\n与 PM2.5 相关性最强的变量:")
        for col, corr_val in correlations[:5]:
            print(f"{col}: {corr_val:.4f}" if corr_val is not None else f"{col}: NaN")

def main():
    """
    主函数
    """
    # 文件路径
    file_path = "PRSA_Data_Changping_20130301-20170228.csv"
    
    # 加载数据
    df = load_data(file_path)
    if df is None:
        return
    
    # 探索数据
    explore_data(df)
    
    # 数据预处理
    df = preprocess_data(df)
    
    # 污染物分析
    analyze_pollutants(df)
    
    # 时间模式分析
    analyze_temporal_patterns(df)
    
    # 相关性分析
    correlation_analysis(df)
    
    # 数据可视化
    visualize_data(df)
    
    print("\n=== 分析完成 ===")
    print("感谢使用 Polars 进行数据分析！")

if __name__ == "__main__":
    main()