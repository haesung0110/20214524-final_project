# 20214524이해성 오픈소스final_project
___
## 1) 프로젝트 개요:
scikit-learn 라이브러리를 사용하여 분류 알고리즘 모델을 생성한 뒤, 뇌 종양 MRI 데이터를 학습하고 새로운 데이터에 대한 예측을 한다.  
이미지 데이터 분류를 위해 knn 알고리즘을 사용하였다. 


## 2) training data set:

```py
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)
```


  
1. 이미지 데이터의 label은 'glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor' 4가지로 분류된다.
2. tumor_dataset/Training 폴더에 들어있는 이미지를 불러와 size를 (64,64)로 조정하고 흑백이미지로 변환한 뒤,  images 배열에 이미지를 추가하고, y 배열에 이미지에 해당하는 label을 추가한다.  
3. images배열을 numpy 배열로 변환한다.  
4. X,y에 각각 image 사이즈를 변환한 images, numpy배열로 변환한 y배열을 할당한다.  

- 데이터 예시1 (glioma_tumor)  
![gg (1)](https://github.com/haesung0110/20214524-final_project/assets/147023827/3efe9be3-542a-48d5-8d45-4c337cd23e59)



## 3) 선택한 알고리즘:

```py
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=2, weights='distance',n_jobs=-1)

#X_train, y_train으로 모델 훈련  
knn.fit(X_train, y_train)

#X_test를 사용하여 예측  
y_pred = knn.predict(X_test)

#정확도 계산 및 출력  
print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_pred))
```

* 알고리즘: knn알고리즘을 사용하여 모델을 훈련하고 예측하였다.
* 하이퍼파라미터: 거리 측정 방법으로 'manhatten', 이웃수는 2, 거리에 대한 가중치는 'distance' (거리의 역수)를 사용하였다.

## 4) 정확도 결과:
y_test 데이터로 예측한 결과의 정확도는 다음과 같다.  
Accuracy: 0.9012775842044135 
  




