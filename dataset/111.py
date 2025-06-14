import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 读取数据
train_df = pd.read_csv('../dataset/split/train.csv')
val_df = pd.read_csv('../dataset/split/val.csv')

# 特征和标签
X_train = train_df['text']
y_train = train_df['label']
X_val = val_df['text']
y_val = val_df['label']

# 文本预处理
X_train = X_train.str.lower().str.replace(r'[^\w\s]', '', regex=True) # 统一小写
X_val = X_val.str.lower().str.replace(r'[^\w\s]', '', regex=True) # 去除标点

# 文本向量化（TF-IDF）
vectorizer = TfidfVectorizer(
    stop_words='english',       # 自动移除英文停用词，减少特征维度，提升模型效率
    ngram_range=(1,3),          # 捕获短语组合，提升上下文敏感
    max_features=10000,         # 限制最大特征数为10,000，防止维度爆炸
    min_df=2                    # 忽略低频词
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 逻辑回归模型
model = LinearSVC(
    class_weight='balanced',  # 自动调整类别权重，处理数据不平衡问题
    C=0.5,                   # 正则化强度的倒数，值越小正则化越强（防止过拟合）
    max_iter=2000,           # 最大迭代次数，确保复杂数据也能收敛
    random_state=42,        # 随机种子，保证每次运行结果可复现
    penalty='l2',           # 使用L2正则化（欧式距离）约束模型复杂度
    loss='squared_hinge'    # 使用平方合页损失函数，对异常值更鲁棒
)
model.fit(X_train_vec, y_train)

# 验证集评估
val_pred = model.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_pred)
print(f'Val Acc: {val_acc:.4f}')
print(classification_report(y_val, val_pred))

# 保存模型和向量器
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'lr_model.pkl')
print('模型已保存为lr_model.pkl')