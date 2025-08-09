import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为XGBoost的DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义模型参数
params = {
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 3,  # 类别数
    'eval_metric': 'merror'  # 评价指标：多分类错误率
}

# 训练模型
num_round = 10
model = xgb.train(params, dtrain, num_round)

# 在测试集上预测
y_pred = model.predict(dtest)

print("-+-" * 25)
print(f"当前树有 {len(model.get_dump())} 棵")
# 查看每棵树的具体节点
for leaf in model.get_dump():
    print(leaf)
print("-+-" * 25)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
