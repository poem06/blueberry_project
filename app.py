#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : app.py.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 面向用户端应用
"""
import pandas as pd
import joblib


def get_smart_input():
    print("=" * 55)
    print("蓝莓产量精准预测系统")
    print("=" * 55)

    user_data = {}

    # 默认值字典
    DEFAULTS = {
        'clonesize': 25.0, 'honeybee': 0.38, 'bumbles': 0.28,
        'andrena': 0.49, 'osmia': 0.59, 'RainingDays': 16.0,
        'AverageRainingDays': 0.26, 'MaxOfUpperTRange': 86.0,
        'MinOfLowerTRange': 51.2
    }

    # 辅助函数：处理带默认值的输入
    def get_input(prompt_text, feature_name, is_required=False):
        while True:
            hint = "必填" if is_required else f"选填, 直接回车默认 {DEFAULTS[feature_name]}"
            val = input(f"{prompt_text} [{hint}]: ")

            if val.strip() == "":
                if is_required:
                    print("此项为必填项，请填写！")
                    continue
                else:
                    return DEFAULTS[feature_name]
            try:
                return float(val)
            except ValueError:
                print("格式错误，请输入纯数字！")

    print("\n【核心指标 (必填)】")
    user_data['fruitset'] = get_input("1. 结果率 (例0.5)", 'fruitset', True)
    user_data['seeds'] = get_input("2. 种子数 (例36.0)", 'seeds', True)
    user_data['fruitmass'] = get_input("3. 果实质量 (例0.45)", 'fruitmass', True)

    print("\n【环境参数 (选填)】")
    user_data['clonesize'] = get_input("4. 克隆大小", 'clonesize')
    user_data['RainingDays'] = get_input("5. 降雨天数", 'RainingDays')
    user_data['MaxOfUpperTRange'] = get_input("6. 花期极端最高气温", 'MaxOfUpperTRange')
    user_data['MinOfLowerTRange'] = get_input("7. 花期极端最低气温", 'MinOfLowerTRange')

    print("\n【蜜蜂密度 (选填)】")
    user_data['honeybee'] = get_input("8. 蜜蜂密度", 'honeybee')
    user_data['bumbles'] = get_input("9. 大黄蜂密度", 'bumbles')
    user_data['andrena'] = get_input("10. 矿蜂密度", 'andrena')
    user_data['osmia'] = get_input("11. 壁蜂密度", 'osmia')

    print("\n正在计算...")

    # 1. 气温矩阵推算 (根据极端最高/最低温，推算上下限与平均值)
    max_t = user_data['MaxOfUpperTRange']
    min_t = user_data['MinOfLowerTRange']
    user_data['MinOfUpperTRange'] = max_t - 27.8  # 推算最高温下限
    user_data['AverageOfUpperTRange'] = max_t - 14.1  # 推算最高温平均
    user_data['MaxOfLowerTRange'] = min_t + 26.2  # 推算最低温上限
    user_data['AverageOfLowerTRange'] = min_t + 13.1  # 推算最低温平均

    # 2. 降雨概率推算
    user_data['AverageRainingDays'] = DEFAULTS['AverageRainingDays']

    user_data['total_bees'] = (user_data['honeybee'] + user_data['bumbles'] +
                               user_data['andrena'] + user_data['osmia'])

    return pd.DataFrame([user_data])


if __name__ == "__main__":
    # 获取用户输入并完成后台推算
    input_df = get_smart_input()

    # 包含了全部 17 个特征，满足模型刁钻的胃口
    expected_columns = [
        'clonesize', 'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
        'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
        'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds',
        'total_bees'
    ]
    processed_df = input_df[expected_columns]

    print("\n>>> 正在预测...")
    try:
        # 加载训练好的模型
        model = joblib.load("saved_models/best_model.pkl")
        prediction = model.predict(processed_df)[0]

        print(f"预测成功！您今年的预计年产量为：{prediction:.2f}")
    except Exception as e:
        print(f"模型加载或预测失败。错误信息: {e}")
