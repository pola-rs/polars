#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polars 中文使用案例 - 北京昌平区空气质量数据分析
==================================================

本脚本演示了如何使用 Polars 对北京昌平区的空气质量数据进行分析。
这是 Polars 中文教程的一个完整示例。

作者: Polars 中文社区
创建日期: 2025年
"""

import polars as pl
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 读取数据
    print("=== 第一步：读取数据 ===")
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
    
    # 使用 Polars 读取 CSV 数据
    df = pl.read_csv(
        "PRSA_Data_Changping_20130301-20170228.csv", 
        schema_overrides=dtypes,
        null_values=["NA", "NaN", "null", ""]
    )
    
    print(f"数据加载成功！共有 {df.height} 行，{df.width} 列")
    print("列名:", df.columns)
    
    # 2. 基本数据探索
    print("\n=== 第二步：基本数据探索 ===")
    print("前5行数据:")
    print(df.head())
    
    print("\n数据类型:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  {col}: {dtype}")
    
    print("\n缺失值统计:")
    null_counts = df.null_count()
    for col, count in zip(df.columns, null_counts.row(0)):
        if count > 0:
            print(f"  {col}: {count} 个缺失值")
    
    # 3. 数据处理和分析
    print("\n=== 第三步：数据处理和分析 ===")
    
    # 计算各污染物的平均值
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    print("各污染物平均浓度:")
    for pollutant in pollutants:
        if pollutant in df.columns:
            avg_value = df.select(pl.col(pollutant).mean()).item()
            print(f"  {pollutant}: {avg_value:.2f} μg/m³")
    
    # 按年份统计平均 PM2.5
    print("\n年度平均 PM2.5 浓度:")
    yearly_pm25 = df.group_by("year").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("year")
    print(yearly_pm25)
    
    # 4. 数据可视化
    print("\n=== 第四步：数据可视化 ===")
    
    # 创建简单的图表
    plt.figure(figsize=(12, 8))
    
    # 年度趋势图
    plt.subplot(2, 2, 1)
    years = yearly_pm25["year"].to_list()
    avg_pm25 = yearly_pm25["avg_PM2.5"].to_list()
    plt.plot(years, avg_pm25, marker='o')
    plt.title("PM2.5 年度平均浓度趋势")
    plt.xlabel("年份")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True, alpha=0.3)
    
    # 各污染物平均浓度
    plt.subplot(2, 2, 2)
    avg_values = []
    valid_pollutants = []
    for pollutant in pollutants:
        if pollutant in df.columns:
            avg_value = df.select(pl.col(pollutant).mean()).item()
            avg_values.append(avg_value)
            valid_pollutants.append(pollutant)
    
    bars = plt.bar(valid_pollutants, avg_values)
    plt.title("各污染物平均浓度")
    plt.ylabel("浓度 (μg/m³)")
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, avg_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # 月度变化模式
    plt.subplot(2, 2, 3)
    monthly_data = df.group_by("month").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("month")
    
    months = monthly_data["month"].to_list()
    monthly_avg = monthly_data["avg_PM2.5"].to_list()
    plt.plot(months, monthly_avg, marker='s')
    plt.title("PM2.5 月度变化模式")
    plt.xlabel("月份")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True, alpha=0.3)
    
    # 小时变化模式
    plt.subplot(2, 2, 4)
    hourly_data = df.group_by("hour").agg(
        pl.col("PM2.5").mean().alias("avg_PM2.5")
    ).sort("hour")
    
    hours = hourly_data["hour"].to_list()
    hourly_avg = hourly_data["avg_PM2.5"].to_list()
    plt.plot(hours, hourly_avg, marker='d', color='green')
    plt.title("PM2.5 小时变化模式")
    plt.xlabel("小时")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("polars_changping_analysis_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("分析完成！图表已保存为 'polars_changping_analysis_demo.png'")
    
    # 5. Polars 特性展示
    print("\n=== 第五步：Polars 特性展示 ===")
    
    # 链式操作示例
    print("链式操作示例 - 筛选2016年的数据并计算统计信息:")
    result = (df
              .filter(pl.col("year") == 2016)
              .select([
                  pl.col("PM2.5").mean().alias("mean_pm25"),
                  pl.col("PM2.5").median().alias("median_pm25"),
                  pl.col("PM2.5").max().alias("max_pm25"),
                  pl.col("PM2.5").min().alias("min_pm25")
              ]))
    print(result)
    
    # 条件筛选示例
    print("\n条件筛选示例 - 高污染天数统计 (PM2.5 > 100):")
    high_pollution_days = df.filter(pl.col("PM2.5") > 100)
    print(f"PM2.5超过100μg/m³的记录有 {high_pollution_days.height} 条")
    
    # 聚合操作示例
    print("\n聚合操作示例 - 不同风向下的平均PM2.5:")
    wind_pm25 = df.group_by("wd").agg(
        pl.col("PM2.5").mean().alias("avg_pm2.5")
    ).sort("avg_pm2.5", descending=True)
    print(wind_pm25.head())

if __name__ == "__main__":
    main()