from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt


def plot_gallery(images, titles, n_row=3, n_col=4):
    plt.figure(figsize=(1.8*n_col, 2.4*n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w, = lfw_people.images.shape

X = lfw_people.data
y = lfw_people.target

n_features = X.shape[1]
print(X.shape)

#stampa immagine
plt.imshow(X[0].reshape((h,w)),cmap=plt.cm.gray)
print(lfw_people.target_names[y[0]])
plt.show()

plt.imshow(X[100].reshape((h,w)),cmap=plt.cm.gray)
print(lfw_people.target_names[y[100]])
plt.show()

target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Dimensione totale dataset:")
print("n_campioni: " + str(n_samples))
print("n_feat: " + str(n_features))
print("n_classi: " + str(n_classes))

from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
from time import time
from sklearn.cross_validation import train_test_split
from sklearn import decomposition

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state=1)
t0=time()
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(time() - t0)

n_comp = 75
pca = decomposition.PCA(n_components=n_comp, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

t0 = time()
logreg.fit(X_train_pca, y_train)
y_pred = logreg.predict(X_test_pca)

print(accuracy_score(y_pred, y_test))
print(time() - t0)

eigenfaces = pca.components_.reshape((n_comp, h, w))
eigenfaces_title = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

plot_gallery(eigenfaces, eigenfaces_title)
plt.show()