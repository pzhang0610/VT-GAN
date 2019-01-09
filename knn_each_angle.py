from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import os
import numpy as np

result = np.zeros((11, 11))
angles_gallery = ['000', '018', '036', '054', '072',
                  '090',
                  '108', '126', '144', '162', '180']
angles_probe = angles_gallery
ix = 0
iy = 0
for g_ang in angles_gallery:
    iy = 0
    for p_ang in angles_probe:
        pid = 63
        X = []
        y = []
        for cond in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
            for p in range(pid, 125):
                # path = '../gaitGAN/gei/%03d/%s/%03d-%s-%s.png' % (
                #     p, cond, p, cond, g_ang)
                path1 = './generated/20181018/054/%03d-%s-%s.png' % (p, cond, g_ang)
                path = path1
                if not os.path.exists(path):
                    continue
                if g_ang == '090':
                    img = cv2.imread(path, 0)
                else:
                    img = cv2.imread(path1, 0)
                img = cv2.resize(img, (64, 64))
                img = img.flatten().astype(np.float32)
                X.append(img)
                y.append(p-63)
                # y.append(p)
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int32)
        pca_model = pca.PCA(n_components=int(min(X.shape)*0.2), whiten=False)
        # print(int(min(X.shape)*0.20))
        pca_model.fit(X)
        X = pca_model.transform(X)

        lda_model = LinearDiscriminantAnalysis(n_components=45)
        lda_model.fit(X, y)
        X = lda_model.transform(X)
        nbrs = KNeighborsClassifier(n_neighbors=1, p=2, weights='distance', metric='euclidean')
        nbrs.fit(X, y)

        testX = []
        testy = []
        pid = 63
        for cond in ['nm-05', 'nm-06']:
            for p in range(pid, 125):
                # path = '../gaitGAN/gei/%03d/%s/%03d-%s-%s.png' % (
                #     p, cond, p, cond, p_ang)
                path1 = './generated/20181018/054/%03d-%s-%s.png' % (p, cond, p_ang)
                path = path1
                if not os.path.exists(path):
                    continue
                if p_ang == '090':
                    img = cv2.imread(path, 0)
                else:
                    img = cv2.imread(path1, 0)
                img = cv2.resize(img, (64, 64))
                img = img.flatten().astype(np.float32)
                testX.append(img)
                testy.append(p-63)
                # testy.append(p)

        testX = np.asarray(testX).astype(np.float32)
        tX = pca_model.transform(testX)
        tX = lda_model.transform(tX)
        s = nbrs.score(tX, testy)
        result[ix][iy] = s
        print(s)
        iy += 1
    ix += 1
print(result)
print(np.mean(result))
print(np.mean(result, axis=0))
np.savetxt("view_analysis.csv", result)