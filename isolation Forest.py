import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

n_samples, n_outliers = 120, 10
rng = np.random.RandomState(0)
cluster_1 = 0.4 * rng.randn(n_samples, 2) + np.array([2, 2])
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate(
    [np.ones((2 * n_samples), dtype=int), -np.ones(n_outliers, dtype=int)]
)

scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='rainbow')
handles, labels = scatter.legend_elements()

plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.title("data distribution")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)

# 得到score，Negative scores represent outliers, positive scores represent inliers
y_pre_score_test = clf.decision_function(cluster_1)
print(y_pre_score_test)

# -1为异常， 1为正常， 可以看下这个函数内部，就是score<0的设为-1
y_pre_label_test = clf.predict(cluster_1)
print(y_pre_label_test)

# 通过网格的方式得到location的x和y坐标
xx, yy = np.meshgrid(np.linspace(-6, 6, 60), np.linspace(-6, 6, 60))
# concat x和y 得到输入的坐标
input_location = np.c_[xx.ravel(), yy.ravel()]

Z = clf.decision_function(input_location)
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, color='rainbow')
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
c = plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.legend([b1, b2, c],
           ["train data",
            "test data",
            "outlier"],
           loc="best")
plt.show()

