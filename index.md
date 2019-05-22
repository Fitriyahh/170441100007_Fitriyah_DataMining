# Pengertian <small>K-mean & K-nn</small>

## Apa itu K-mean dan K-nn ?

**K-MEANS**

K-means clustering adalah metode  yang digunakan  pada data yang tidak memiliki label. Arti K-means sendiri yaitu mengelompokkan data-data dengan karakter mirip.  Untuk melakukan clustering kita perlu algoritma yang mendukung untuk pengimplementasian dari metode tersebut, salah satunya K-means. Tujuan dari k-means itu sendiri membagi suatu data dalam beberapa cluster (kelompok) sebanyak *k,* yang dimana jumlah *k-nya* ditentukan oleh kita dan diwakili oleh [***Mean (Rata-rata)***](https://id.wikipedia.org/wiki/Rata-rata). *Mean* dari setiap *cluster* diasumsikan sebagai ringkasan yang baik dari setiap observasi dari *cluster* tersebut.

**Kelebihan**

1. Menggunakan prinsip yang sederhana, dapat dijelaskan dalam non-statistik
2. Sangat fleksibel, dapat dengan mudah diadaptasi
3. Sangat umum digunakan
4. Waktu yang dibutuhkan untuk menjalankan nya relatif cepat

**Kekurangan**

1. Tidak optimal digunakan untuk data yang jumlahnya terlalu banyak sampai bermiliyar.
2. Karena menggunakan k buah acak, tidak di jamin untuk menemukan kumpulan cluster yang optimal
3. dapat terjadinya curse of dimensionality, apabila jarak antara cluster yang satu dengan yang lain memiliki banyak dimesi.



**K-Means Clustering** secara umum dilakukan dengan algoritma sebagai berikut:

1. Menentukan jumlah cluster,
2. Menalokasikan data ke dalam cluster secara random,
3. Menghitung centroid/rata-rata dari data yang ada di masing-masing cluster,
4. Mengalokasikan masing-masing data ke centroid/rata-rata terdekat.



**contohnya sebagai berikut..**



**K-NN**

K-nearest neighbors atau k-nn adalah algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran (*train data sets*), yang diambil dari k tetangga terdekatnya (*nearest neighbors*). Dengan k merupakan banyaknya tetangga terdekat.

##### **Algoritma K-Nearest Neighbors**

1. Tentukan k bilangan bulat positif berdasarkan ketersediaan data pembelajaran.
2. Pilih tetangga terdekat dari data baru sebanyak k.
3. Tentukan klasifikasi paling umum pada langkah (ii), dengan menggunakan frekuensi terbanyak.
4. Keluaran klasifikasi dari data sampel baru.

berikut ini contoh coding program k-nn menggunakan data bunga iris dalam pemrograman python :

```
from sklearn import datasets
import pandas as pd

from sklearn.linear_model import logistic_regression_path

iris=datasets.load_iris()

print(iris.data)
print(iris.target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,test_size=0.33)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

from sklearn.metrics import accuracy_score
print("accuracy is ")
print(accuracy_score(y_test,clf.predict(x_test)))

import matplotlib.pyplot as plt

accuracy_values=[]

for x in range(1,x_train.shape[0]):
    clf=KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
    accuracy=accuracy_score(y_test,clf.predict(x_test))
    accuracy_values.append([x,accuracy])
    pass

**ACCURACY DATA**
import numpy as np
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("accuracy")
plt.show()
```

![](C:\Users\axioo_pc\Documents\mkdocs-material-master\docs\assets\images\dm1.png)



**Referensi**

[1]: https://www.advernesia.com/blog/data-science/pengertian-dan-cara-kerja-algoritma-k-nearest-neighbours-knn/
[2]: https://brotodata.com/2018/06/28/penjelasan-k-means-clustering/

[]: 



# DECISION TREE <small></small>

## Apa itu Decision tree (pohon keputusan) ?

**Pengertian Pohon Keputusan**

Pohon keputusan adalah salah satu metode klasifikasi yang paling populer karena mudah untuk diinterpretasi oleh manusia. Konsep dari pohon keputusan adalah mengubah data menjadi pohon keputusan dan aturan-aturan keputusan. Manfaat utama dari penggunaan pohon keputusan adalah kemampuannya untuk mem-break down proses pengambilan keputusan yang kompleks menjadi lebih simpel sehingga pengambil keputusan akan lebih menginterpretasikan solusi dari permasalahan. Pohon Keputusan juga berguna untuk mengeksplorasi data, menemukan hubungan tersembunyi antara sejumlah calon variabel input dengan sebuah variabel target. Pohon keputusan memadukan antara eksplorasi data dan pemodelan, sehingga sangat bagus sebagai langkah awal dalam proses pemodelan bahkan ketika dijadikan sebagai model akhir dari beberapa teknik lain.

Pada pohon keputusan terdapat tiga jenis node, antara lain :
1. **Akar**
Merupakan node teratas, pada node ini tidak ada input dan dapat tidak mempunyai output atau dapat mempunyai output lebih dari satu.
2. **Internal node**
Merupakan node percabangan, pada node ini hanya terdapat satu input dan mempunyai output minimal dua.
3. **Daun**
Merupakan node akhir atau terminal node, pada node ini hanya terdapat satu input dan tidak mempunyai output (simpul terminal).
[](assets\images\Untitled.png)

Sebagai contoh suatu pohon disusun oleh simpul t1, t2, …, t4 dengan rincian terdapat 3 daun, 1 akar, dan 1 internal node. Setiap pemilah (split) memilah simpul nonterminal menjadi dua simpul yang saling lepas. Hasil prediksi respon suatu amatan terdapat pada simpul terminal (daun).

Konsep dari pohon keputusan adalah mengubah data menjadi pohon keputusan dan aturan-aturan keputusan. Pohon keputusan merupakan himpunan aturan if — then, dimana setiap path dalam pohon dihubungkan dengan sebuah aturan dimana premis terdiri atas sekumpulan node yang ditemui dan kesimpulan dari aturan terdiri atas kelas yang dihubungkan dengan daun dari path. Pembentukan pohon keputusan terdiri dari beberapa tahap :

1. Konstruksi pohon diawali dengan pembentukan akar (terletak paling atas). Kemudian data dibagi berdasarkan atribut-atribut yang cocok untuk dijadikan daun.

2. Pemangkasan pohon (tree pruning) yaitu mengidentifikasikan dan membuang cabang yang tidak diperlukan pada pohon yang telah terbentuk. Hal ini dikarenakan pohon keputusan yang dikontruksi dapat berukuran besar, maka dapat disederhanakan dengan melakukan pemangkasan berdasarkan nilai kepercayaan (confident level). Pemangkasan pohon dilakukan selain untuk pengurangan ukuran pohon juga bertujuan untuk mengurangi tingkat kesalahan prediksi pada kasus baru dari hasil pemecahan yang dilakukan dengan divide and conquer. Pruning ada dua pendekatan yaitu :

a. **Pre-pruning** yaitu menghentikan pembangunan suatu subtree lebih awal (dengan memutuskan untuk tidak lebih jauh mempartisi data training). Saat seketika berhenti, maka node berubah menjadi leaf (node akhir). Node akhir ini menjadi kelas yang paling sering muncul di antara subset sampel.

b. **Post-pruning** yaitu menyederhanakan tree dengan cara membuang beberapa cabang subtree setelah tree selesai dibangun. Node yang jarang dipotong akan menjadi leaf (node akhir) dengan kelas yang paling sering muncul.

3. Pembentukan aturan keputusan yaitu membuat aturan keputusan dari pohon yang telah dibentuk. Aturan tersebut dapat dalam bentuk if — then diturunkan dari pohon keputusan dengan melakukan penelusuran dari akar sampai ke daun. Untuk setiap simpul dan percabangannya akan diberikan di if, sedangkan nilai pada daun akan ditulis di then. Setelah semua aturan dibuat maka aturan dapat disederhanakan atau digabung. 

**Kelebihan**

1. Menggunakan prinsip yang sederhana, dapat dijelaskan dalam non-statistik
2. Sangat fleksibel, dapat dengan mudah diadaptasi
3. Sangat umum digunakan
4. Waktu yang dibutuhkan untuk menjalankan nya relatif cepat

**Kelebihan**
1. Daerah pengambilan keputusan yang sebelumnya kompleks dan sangat global, dapat diubah menjadi lebih simpel dan spesifik.
2. Eliminasi perhitungan-perhitungan yang tidak diperlukan, karena ketika menggunakan metode pohon keputusan maka sample diuji hanya berdasarkan kriteria atau kelas tertentu.
3. Fleksibel untuk memilih fitur dari internal node yang berbeda, fitur yang terpilih akan membedakan suatu kriteria dibandingkan kriteria yang lain dalam node yang sama. 
4. Kefleksibelan metode pohon keputusan ini meningkatkan kualitas keputusan yang dihasilkan jika dibandingkan ketika menggunakan metode penghitungan satu tahap yang lebih konvensional
Dalam analisis multivariat, dengan kriteria dan kelas yang jumlahnya sangat banyak, seorang penguji biasanya perlu untuk mengestimasikan baik itu distribusi dimensi tinggi ataupun parameter tertentu dari distribusi kelas tersebut. Metode pohon keputusan dapat menghindari munculnya permasalahan ini dengan menggunakan criteria yang jumlahnya lebih sedikit pada setiap node internal tanpa banyak mengurangi kualitas keputusan yang dihasilkan.

**Kekurangan**
1. Terjadi overlap terutama ketika kelas-kelas dan criteria yang digunakan jumlahnya sangat banyak. Hal tersebut juga dapat menyebabkan meningkatnya waktu pengambilan keputusan dan jumlah memori yang diperlukan.
2. Pengakumulasian jumlah eror dari setiap tingkat dalam sebuah pohon keputusan yang besar.
3. Kesulitan dalam mendesain pohon keputusan yang optimal.
Hasil kualitas keputusan yang didapatkan dari metode pohon keputusan sangat tergantung pada bagaimana pohon tersebut didesain.

berikut codingan decision tree :
kita harus download beberapa library dari python antaralain **pandas** dan **scikit-learn**

```
# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv("LasVegasTripAdvisorReviews-Dataset.csv",sep= ',', header = 1)

    # Printing the dataset shape
    print ("Dataset Lenght: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset observations
    print('dataset :')
    print (balance_data.head())
    return balance_data

# Function to split the dataset
def splitdataset(balance_data):

    # Seperating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 100)

    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))

    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

    print("Report : ",
    classification_report(y_test, y_pred))

# Driver code
def main():

    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


# Calling main function
if __name__=="__main__":
    main()

```


**Referensi**

[1]: https://www.academia.edu/31573614/Tentang_Pohon_Keputusan_Decision_Tree
[2]: https://fairuzelsaid.wordpress.com/2009/11/24/data-mining-konsep-pohon-keputusan/
[3]:https://medium.com/@mimubarok.mim/decision-tree-pohon-keputusan-6484ad30c289



# DECISION TREE <small></small>

## Apa itu Decision tree (pohon keputusan) ?

**Pengertian Pohon Keputusan**

Pohon keputusan adalah salah satu metode klasifikasi yang paling populer karena mudah untuk diinterpretasi oleh manusia. Konsep dari pohon keputusan adalah mengubah data menjadi pohon keputusan dan aturan-aturan keputusan. Manfaat utama dari penggunaan pohon keputusan adalah kemampuannya untuk mem-break down proses pengambilan keputusan yang kompleks menjadi lebih simpel sehingga pengambil keputusan akan lebih menginterpretasikan solusi dari permasalahan. Pohon Keputusan juga berguna untuk mengeksplorasi data, menemukan hubungan tersembunyi antara sejumlah calon variabel input dengan sebuah variabel target. Pohon keputusan memadukan antara eksplorasi data dan pemodelan, sehingga sangat bagus sebagai langkah awal dalam proses pemodelan bahkan ketika dijadikan sebagai model akhir dari beberapa teknik lain.

Pada pohon keputusan terdapat tiga jenis node, antara lain :

1. **Akar**
   Merupakan node teratas, pada node ini tidak ada input dan dapat tidak mempunyai output atau dapat mempunyai output lebih dari satu.
2. **Internal node**
   Merupakan node percabangan, pada node ini hanya terdapat satu input dan mempunyai output minimal dua.
3. **Daun**
   Merupakan node akhir atau terminal node, pada node ini hanya terdapat satu input dan tidak mempunyai output (simpul terminal).
   [](assets\images\Untitled.png)

Sebagai contoh suatu pohon disusun oleh simpul t1, t2, …, t4 dengan rincian terdapat 3 daun, 1 akar, dan 1 internal node. Setiap pemilah (split) memilah simpul nonterminal menjadi dua simpul yang saling lepas. Hasil prediksi respon suatu amatan terdapat pada simpul terminal (daun).

Konsep dari pohon keputusan adalah mengubah data menjadi pohon keputusan dan aturan-aturan keputusan. Pohon keputusan merupakan himpunan aturan if — then, dimana setiap path dalam pohon dihubungkan dengan sebuah aturan dimana premis terdiri atas sekumpulan node yang ditemui dan kesimpulan dari aturan terdiri atas kelas yang dihubungkan dengan daun dari path. Pembentukan pohon keputusan terdiri dari beberapa tahap :

1. Konstruksi pohon diawali dengan pembentukan akar (terletak paling atas). Kemudian data dibagi berdasarkan atribut-atribut yang cocok untuk dijadikan daun.
2. Pemangkasan pohon (tree pruning) yaitu mengidentifikasikan dan membuang cabang yang tidak diperlukan pada pohon yang telah terbentuk. Hal ini dikarenakan pohon keputusan yang dikontruksi dapat berukuran besar, maka dapat disederhanakan dengan melakukan pemangkasan berdasarkan nilai kepercayaan (confident level). Pemangkasan pohon dilakukan selain untuk pengurangan ukuran pohon juga bertujuan untuk mengurangi tingkat kesalahan prediksi pada kasus baru dari hasil pemecahan yang dilakukan dengan divide and conquer. Pruning ada dua pendekatan yaitu :

a. **Pre-pruning** yaitu menghentikan pembangunan suatu subtree lebih awal (dengan memutuskan untuk tidak lebih jauh mempartisi data training). Saat seketika berhenti, maka node berubah menjadi leaf (node akhir). Node akhir ini menjadi kelas yang paling sering muncul di antara subset sampel.

b. **Post-pruning** yaitu menyederhanakan tree dengan cara membuang beberapa cabang subtree setelah tree selesai dibangun. Node yang jarang dipotong akan menjadi leaf (node akhir) dengan kelas yang paling sering muncul.

3. Pembentukan aturan keputusan yaitu membuat aturan keputusan dari pohon yang telah dibentuk. Aturan tersebut dapat dalam bentuk if — then diturunkan dari pohon keputusan dengan melakukan penelusuran dari akar sampai ke daun. Untuk setiap simpul dan percabangannya akan diberikan di if, sedangkan nilai pada daun akan ditulis di then. Setelah semua aturan dibuat maka aturan dapat disederhanakan atau digabung. 

**Kelebihan**

1. Menggunakan prinsip yang sederhana, dapat dijelaskan dalam non-statistik
2. Sangat fleksibel, dapat dengan mudah diadaptasi
3. Sangat umum digunakan
4. Waktu yang dibutuhkan untuk menjalankan nya relatif cepat

**Kelebihan**

1. Daerah pengambilan keputusan yang sebelumnya kompleks dan sangat global, dapat diubah menjadi lebih simpel dan spesifik.
2. Eliminasi perhitungan-perhitungan yang tidak diperlukan, karena ketika menggunakan metode pohon keputusan maka sample diuji hanya berdasarkan kriteria atau kelas tertentu.
3. Fleksibel untuk memilih fitur dari internal node yang berbeda, fitur yang terpilih akan membedakan suatu kriteria dibandingkan kriteria yang lain dalam node yang sama. 
4. Kefleksibelan metode pohon keputusan ini meningkatkan kualitas keputusan yang dihasilkan jika dibandingkan ketika menggunakan metode penghitungan satu tahap yang lebih konvensional
   Dalam analisis multivariat, dengan kriteria dan kelas yang jumlahnya sangat banyak, seorang penguji biasanya perlu untuk mengestimasikan baik itu distribusi dimensi tinggi ataupun parameter tertentu dari distribusi kelas tersebut. Metode pohon keputusan dapat menghindari munculnya permasalahan ini dengan menggunakan criteria yang jumlahnya lebih sedikit pada setiap node internal tanpa banyak mengurangi kualitas keputusan yang dihasilkan.

**Kekurangan**

1. Terjadi overlap terutama ketika kelas-kelas dan criteria yang digunakan jumlahnya sangat banyak. Hal tersebut juga dapat menyebabkan meningkatnya waktu pengambilan keputusan dan jumlah memori yang diperlukan.
2. Pengakumulasian jumlah eror dari setiap tingkat dalam sebuah pohon keputusan yang besar.
3. Kesulitan dalam mendesain pohon keputusan yang optimal.
   Hasil kualitas keputusan yang didapatkan dari metode pohon keputusan sangat tergantung pada bagaimana pohon tersebut didesain.

berikut codingan decision tree :
kita harus download beberapa library dari python antaralain **pandas** dan **scikit-learn**

```
# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv("LasVegasTripAdvisorReviews-Dataset.csv",sep= ',', header = 1)

    # Printing the dataset shape
    print ("Dataset Lenght: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset observations
    print('dataset :')
    print (balance_data.head())
    return balance_data

# Function to split the dataset
def splitdataset(balance_data):

    # Seperating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 100)

    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))

    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

    print("Report : ",
    classification_report(y_test, y_pred))

# Driver code
def main():

    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


# Calling main function
if __name__=="__main__":
    main()

```

**Referensi**

[1]: https://www.academia.edu/31573614/Tentang_Pohon_Keputusan_Decision_Tree
[2]: https://fairuzelsaid.wordpress.com/2009/11/24/data-mining-konsep-pohon-keputusan/
[3]: https://medium.com/@mimubarok.mim/decision-tree-pohon-keputusan-6484ad30c289