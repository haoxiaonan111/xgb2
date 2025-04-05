# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st  

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib  

# 导入 NumPy 库，用于数值计算
import numpy as np  

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd  

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap  

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt  

# 从 LIME 库中导入 LimeTabularExplainer，用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer  

# 加载训练好的随机森林模型（RF.pkl）
model = joblib.load('xgb.pkl')  

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')  

# 定义特征名称，对应数据集中的列名
feature_names = [  
    "Medical_social_support",       # 医疗社会支持  
    "Age",       # 年龄  
    "Serum_creatinine",        # 血清肌酐  
    "Total_cholesterol",  # 总胆固醇  
    "Rating_of_mobility",      # 活动性评级  
    "Ejection_fraction",       # 射血分数  
    "Depression_symptoms",   # 抑郁症状  
]  
# Streamlit 用户界面
st.title("Prediction of HAD in older patients with acute heart failure")  # 设置网页标题

# 医疗社会支持：数值输入框
Medical_social_support = st.number_input("Medical social support:", min_value=19, max_value=95, value=43)  

# 年龄：数值输入框
Age = st.number_input("Age:", min_value=60, max_value=120, value=79)  

# 血清肌酐：数值输入框
Serum_creatinine = st.number_input("Serum creatinine:", min_value=1, max_value=10000, value=60)  

# 总胆固醇：数值输入框
Total_cholesterol = st.number_input("Total cholesterol:", min_value=0.01, max_value=100.00, value=5.65) 

# Rating of mobility：分类选择框（0-12）
Rating_of_mobility = st.selectbox("Rating of mobility:", options=[0, 2, 4, 6, 8, 10, 12])  

# 射血分数：数值输入框
Ejection_fraction = st.number_input("Ejection fraction:", min_value=1, max_value=100, value=28)  

# 抑郁症状：数值输入框
Depression_symptoms = st.number_input("Depression symptoms:", min_value=0, max_value=21, value=4)  

# 处理输入数据并进行预测
feature_values = [Medical_social_support, Age, Serum_creatinine, Total_cholesterol, Rating_of_mobility, Ejection_fraction, Depression_symptoms]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
     # 预测类别（0：无had，1：有had）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 对概率值进行格式化，保留三位小数
    formatted_proba = ["{:.3f}".format(prob) for prob in predicted_proba]
    proba_str = ", ".join(formatted_proba)

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: HAD risk, 0: No HAD risk)")
    st.write(f"**Prediction Probabilities:** [{proba_str}]")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of HAD. "
            f"The model predicts that your probability of developing HAD is {probability:.1f}%. "
     )
    else:
        advice = (
            f"According to our model, you have a low risk of HAD. "
            f"The model predicts that your probability of not developing HAD is {probability:.1f}%. "
        )
    # 显示建议
    st.write(advice)
  
# SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.Explainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap(features)

    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value, shap_values.values[0], pd.DataFrame(features, columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        shap.force_plot(explainer_shap.expected_value, shap_values.values[0], pd.DataFrame(features, columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')