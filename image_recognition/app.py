import streamlit as st
from PIL import Image
from model_utils import ImageClassifier
import time

# 设置页面配置
st.set_page_config(
    page_title="智能识物 - AI 图像分类器",
    page_icon="🖼️",
    layout="wide"
)

# 加载模型并缓存，避免每次刷新页面都重新加载
@st.cache_resource
def load_model():
    return ImageClassifier()

# 侧边栏
with st.sidebar:
    st.title("💡 关于此应用")
    st.markdown("""
    此应用基于 **PyTorch** 和 **ResNet50** 预训练模型构建。
    它可以识别超过 1000 种常见的物体类别（ImageNet）。
    
    ### 如何使用：
    1. 上传一张图片 (PNG, JPG, JPEG)。
    2. 等待 AI 进行推理。
    3. 查看排名前 5 的识别结果及其置信度。
    """)
    st.divider()
    st.info("模型：ResNet50 (预训练于 ImageNet-1K)")

# 主界面
st.title("🖼️ 智能识物 - AI 图像分类器")
st.write("上传一张图片，让 AI 告诉你是啥！")

# 模型加载提示
with st.status("正在初始化 AI 模型...", expanded=False) as status:
    classifier = load_model()
    status.update(label="模型已就绪!", state="complete", expanded=False)

# 图片上传器
uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示图片
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" 上传的图片")
        image = Image.open(uploaded_file)
        st.image(image, caption='已上传图片', use_container_width=True)
        
    with col2:
        st.subheader("AI 识别结果")
        with st.spinner('正在分析中...'):
            # 执行推理
            start_time = time.time()
            results = classifier.predict(image)
            end_time = time.time()
            
            # 显示预测时间
            st.caption(f"推理用时: {end_time - start_time:.3f} 秒")
            
            # 显示结果
            for i, res in enumerate(results):
                # 格式化类别名：去除多余的连字符
                category = res['category'].replace('_', ' ').title()
                prob = res['probability']
                
                # 第一名高亮显示
                if i == 0:
                    st.success(f"**Top 1: {category}** ({prob:.2%})")
                else:
                    st.write(f"{i+1}. {category} ({prob:.2%})")
                
                # 进度条展示概率
                st.progress(prob)
                
    # 底部反馈
    st.divider()
    st.markdown("**项目说明：** 这是一个典型的全流程 AI 实战项目，涵盖了模型加载、图像预处理、推理计算以及 Web 应用部署。")
else:
    # 未上传图片时的占位符
    st.info("请在上方上传图片以开始识别。")
    
    # 示例图片推荐
    st.write("还没有图片？可以去网上搜一些猫、狗、汽车或水果的图片试试看！")
