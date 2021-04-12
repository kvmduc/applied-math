# TOÁN ỨNG DỤNG VÀ THỐNG KÊ
Course `MTH00051`: Applied Mathematics and Statistic
Class 18CLC4 Term III / 2019-2020 

Author:
-   `18127080`: Kiều Vũ Minh Đức (Github: [@kvmduc](https://github.com/kvmduc))

---

# PROJECT 1 : Color Compression

## Ý tưởng thực hiện:

Khi máy tính đọc vào một bức ảnh màu, máy tính sẽ nhận diện đó là một ma trận (tùy thuộc vào kích thước ảnh và số lượng kênh màu, ta có vô số kích cỡ ma trận khác nhau) với mỗi phần tử được lưu trữ dưới những giá trị của những kênh màu RGB, giá trị thuộc đoạn [0,255]. Ý tưởng thực hiện việc nén màu ảnh dựa vào thuật toán học máy **K-means clustering** (một thuật toán unsupervised learning cơ bản) để gom cụm những pixel lại với nhau, tạo thành một cluster. Một điểm đặc trưng của **K-means clustering** đó là chúng ta không biết những **data point** cho trước (trong bài toán này là các pixel) được gom cụm dựa vào tiêu chí nào, thuật toán chỉ trả về **k cluster** được đưa vào. Do đó chúng ta có thể hoàn toàn chủ động việc chọn số lượng **k** tùy ý để giảm số lượng màu mà ảnh phải thể hiện.

### Giới thiệu các bước thực hiện của thuật toán :

Thuật toán **K-means clustering** có thứ tự hoạt động như sau:

+ <u>Bước 1</u> : Khởi tạo **k centroid**
+ <u>Bước 2</u> : Thực hiện việc phân cụm (gán **label**) của từng pixel dựa vào các **centroid**
+ <u>Bước 3</u> : Cập nhật **centroid** mới, giá trị của **centroid<sub>i</sub>** mới bằng giá trị trung bình (means,median,...) của cluster đó **i**
+ <u>Bước 4</u> : Thực hiện lặp lại cho đến khi kết quả hội tụ.

### Mô tả các hàm trong chương trình :

```python
def kmeans(img_1d, k_clusters, max_iter = 1000, init_centroids='in_pixels'):
    #flatten array
    row = img_1d.shape[0]
    column = img_1d.shape[1]
    channel = img_1d.shape[2]
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    #init centroids
    centroid = [init_centroid(img_1d, k_clusters, init_centroids)]
    label = []
    while True and max_iter:
        #assign label of datapoint
        new_label = assign_label(img_1d, centroid[-1])
        label.append(new_label)
        new_centroid = update_centroid(img_1d,label[-1],k_clusters, channel)
        if converge_check(centroid[-1],new_centroid) :
            break
        max_iter-=1
        centroid.append(new_centroid)
    new_img = update_data_point(img_1d,k_clusters,label[-1],centroid[-1])
    new_img = new_img.reshape(row,column,channel)
    return centroid[-1], new_img
```

1. Ở bước đầu tiên, việc xử lý một ma trận nhiều chiều khá phức tạp (với ảnh RGB khi đọc vào sẽ có số chiều là 3), do đó chúng ta cần thực hiện việc chuyển ma trận ảnh về 2 chiều như trên. sau đó thực hiện việc khởi tạo **centroid**. Tiếp đến thực hiện thuật toán trên lần lượt theo một số lượng lần `max_iter` cho trước (Ở đây là 1000 lần), hoặc đến khi kết quả hội tụ (điều kiện hội tụ được thể hiện bên dưới).

```python
def init_centroid(img_1d, k_cluster, init_centroid_type = 'in_pixels'):
    if init_centroid_type == 'in_pixels':
        return img_1d[np.random.choice(img_1d.shape[0], k_cluster, replace= False)]
    if init_centroid_type == 'random':
        return np.random.choice(256, size = (k_cluster, img_1d.shape[1]), replace=False)
```

2. Ở bước thứ hai, chúng ta sẽ khởi tạo **k centroid**, do yêu cầu thực hiện 2 cách khởi tạo khác nhau, ta có như sau :
   + Đối kiểu khởi tạo là `random`, chúng ta sẽ return **k centroid** với mỗi giá trị là integer thuộc đoạn [0,255], và mỗi centroid có số channel của một pixel.
   + Đối kiểu khởi tạo là `in_pixels`, chúng ta sẽ return **k centroid** với mỗi giá trị thuộc ảnh
   
   Có thể sử dụng hàm `numpy.random.choice(replace = False)` để không bị trùng giá trị centroid.

```python
def assign_label(img_1d, centroid):
    #norm-2 between pixel and centroid
    distance = np.sqrt(np.sum((img_1d - centroid[0]) ** 2, axis=1))
    distance = distance.reshape((img_1d.shape[0], 1))
    for i in range(1,centroid.shape[0]):
        temp = np.sqrt(np.sum((img_1d - centroid[i]) ** 2, axis=1))
        temp = temp.reshape((img_1d.shape[0], 1))
        distance = np.concatenate((distance,temp),axis=1)
    #return smallest distance's label centroid for each pixel
    return np.argmin(distance,axis = 1)
```

3. Ở bước thứ ba, với mỗi **centroid**, chúng ta sẽ thực hiện việc tính độ chênh lệch giữa toàn bộ **pixel** đến **centroid** và lưu lại ở một vector (có kích thước là $Count(pixel) * 1$ ), lặp lại với **k centroid** và ghép các vector lại với nhau. 
   Sau cùng hàm `numpy.argmin()` sẽ trả về **index** của **centroid** làm cho độ chênh lệch ít nhất, nói cách khác là **centroid** gần **pixel** đó nhất.

```python
def update_centroid(img_1d, label, k_cluster, channel):
    centroid = np.zeros((k_cluster,channel))
    for k in range(k_cluster):
        #slice cluster k from img_1d
        cluster_k = img_1d[label == k, :]
        #if cluster have 0 data point -> pass update centroid
        if len(cluster_k) == 0:
            continue
        centroid[k:] = np.mean(cluster_k, axis = 0)
    return centroid
```

4. Ở bước thứ tư, chúng ta sẽ cập nhật lại **centroid** mới từ một **cluster** đã được tìm ra, với mỗi giá trị **label k** , chúng ta sẽ trả ra những **pixel** được gán **label k** trong ma trận ảnh, sau đó tính giá trị của **centroid** mới bằng hàm `numpy.mean`() hoặc `numpy.median()` hoặc `numpy.average()` của những **pixel** có **label k** đó.
   Nếu xảy ra trường hợp **cluster k** không có một **pixel** nào thì có thể bỏ qua việc cập nhật **cluster** đó 

   ```python
   def converge_check(centroid, new_centroid):
       #if distance of value RGB <= Epsilon between old_centroid va new_centroid, assume program has converged
       E = 2
       return np.allclose(centroid, new_centroid, atol = E, equal_nan = True)
   ```

5. Ở bước tiếp theo,  chúng ta sẽ xét giá trị của **centroid** của 2 lần tính toán gần nhất, nếu khoảng cách giữa các giá trị của **centroid** lần tính trước và **centroid** vừa tính hiện tại nhỏ hơn một giá trị **epsion E** (ở đây em chọn bằng 2, do mỗi màu chênh lệch nhau 2 đơn vị là đủ lớn nhưng không thấy được sự chênh lệch) thì giá trị **centroid** vừa tính xem như đủ tốt để dừng chương trình

   ```python
   def update_data_point(img_1d,k_clusters,label,centroids):
       new_img = np.zeros((img_1d.shape[0],img_1d.shape[1]))
       for k in range(k_clusters):
           new_img[label == k, :] += centroids[k]
       return new_img
   ```

6. Ở bước cuối cùng, với mỗi **pixel** có **label k** thì được thay thế bằng giá trị của **centroid** của **cluster k**.

### Kết quả thực nghiệm

Ở đây em chuẩn bị hai bức ảnh RGB như bên dưới:

##### Sample 1:

![img](C:\Users\Casablanca\Downloads\img.jpg)

##### Sample 2:

![95821225_312246226446160_6256357643221204992_n](D:\Download\Picture\95821225_312246226446160_6256357643221204992_n.jpg)

#### Với k =  3, các sample sẽ trở thành  :

##### Sample 1:

![image-20200810221152692](C:\Users\Casablanca\AppData\Roaming\Typora\typora-user-images\image-20200810221152692.png)

So sánh kết quả centroid của chương trình so với hàm `KMeans()` của `scikit-learn`:

+ Kết quả của chương trình:

  ```
  [[ 27.11074895  23.23795461  37.05021567]
   [132.75402695 108.64256408 127.97712998]
   [213.93582427 197.89459377 196.99130479]]
  ```

+ Kết quả của hàm `KMeans()`:

  ```
  [[216.78914298 201.72090305 199.70124936]
   [ 27.9102496   23.87159031  37.78548496]
   [139.13654026 114.35678839 132.96341292]]
  ```

##### Sample 2:

![image-20200810221253337](C:\Users\Casablanca\AppData\Roaming\Typora\typora-user-images\image-20200810221253337.png)

So sánh kết quả centroid của chương trình so với hàm `KMeans()` của `scikit-learn`:

+ Kết quả của chương trình:

  ```
  [[ 99.16591958  73.01802968  71.89929276]
   [214.42588401 189.78028021 164.27015719]
   [231.09701827 227.80092505 235.5129304 ]]
  ```

+ Kết quả của hàm `KMeans()`:

  ```
  [[214.21397928 189.36706497 163.55671079]
   [ 98.18934791  72.11360875  71.06088835]
   [230.98178257 227.61155484 235.25117549]]
  ```

#### Với k = 5, các sample sẽ trở thành :

##### Sample 1:

![image-20200810222822162](C:\Users\Casablanca\AppData\Roaming\Typora\typora-user-images\image-20200810222822162.png)

So sánh kết quả centroid của chương trình so với hàm `KMeans()` của `scikit-learn`:

+ Kết quả của chương trình:

  ```
  [[187.32680386 156.87470442 166.10551543]
   [ 78.76545002  64.02606111  84.45174893]
   [225.03273586 217.46882044 211.82660728]
   [ 23.86101242  20.70780019  33.97144384]
   [131.58884132 107.77308001 128.73352065]]
  ```

+ Kết quả của hàm `KMeans()`:

  ```
  [[203.73453393 171.85772886 176.30985493]
   [ 25.12415836  21.68168469  35.21490772]
   [ 97.32235961  79.39268418 100.85620814]
   [226.16286964 226.14607174 218.9672091 ]
   [152.0075481  125.53107437 143.45382943]]
  ```

##### Sample 2:

![image-20200810222851406](C:\Users\Casablanca\AppData\Roaming\Typora\typora-user-images\image-20200810222851406.png)

So sánh kết quả centroid của chương trình so với hàm `KMeans()` của `scikit-learn`:

+ Kết quả của chương trình:

  ```
  [[217.48920464 191.36584311 159.90620641]
   [161.81931492 122.99901363 116.33384146]
   [234.53692855 232.5937417  240.79684289]
   [ 71.1076808   55.44172631  54.9384147 ]
   [219.08298268 209.86360201 213.26038688]]
  ```

+ Kết quả của hàm `KMeans()`:

  ```
  [[234.42981708 232.4562339  240.65962771]
   [155.08618135 116.84284868 111.52350849]
   [ 68.63251281  53.77586887  53.27713626]
   [216.81702221 190.20300392 158.88677286]
   [218.96910408 209.41376421 212.20920134]]
  ```

#### Với k = 7, các sample sẽ trở thành :

##### Sample 1:

![image-20200810223205043](C:\Users\Casablanca\AppData\Roaming\Typora\typora-user-images\image-20200810223205043.png)

So sánh kết quả centroid của chương trình so với hàm `KMeans()` của `scikit-learn`:

+ Kết quả của chương trình:

  ```
  [[ 87.42918187  71.05190065  91.61995905]
   [226.11508453 227.09781823 219.72214998]
   [ 39.25553437  33.0371552   52.39796911]
   [165.56514552 138.6582678  154.17332064]
   [125.85993683 103.05010933 124.93179057]
   [ 18.73150964  16.53337238  27.31817581]
   [209.01321185 175.88754746 178.61169324]]
  ```

+ Kết quả của hàm `KMeans()`:

  ```
  [[ 18.74285499  16.54468109  27.340678  ]
   [220.25028318 188.48989283 188.62790799]
   [137.2222861  112.28641614 132.63727591]
   [180.30142042 152.09642535 163.58752092]
   [ 40.0320744   33.63093566  52.97509964]
   [ 94.08587417  76.63971591  97.87904441]
   [225.66871563 233.69071498 224.3926145 ]]
  ```

#### Sample 2:

![image-20200810223330136](C:\Users\Casablanca\AppData\Roaming\Typora\typora-user-images\image-20200810223330136.png)

So sánh kết quả centroid của chương trình so với hàm `KMeans()` của `scikit-learn`:

+ Kết quả của chương trình:

  ```
  [[236.0065444  234.57176253 242.6368169 ]
   [224.03711266 218.33159999 226.16062656]
   [146.36855498 107.13220236 104.04919721]
   [224.52372103 199.58612538 163.37368824]
   [203.89826205 171.96878694 144.42658582]
   [213.33281318 199.74182192 197.486936  ]
   [ 66.95346244  52.5588615   52.02922535]]
  ```

+ Kết quả của hàm `KMeans()`:

  ```
  [[222.9321937  196.62669236 158.17419034]
   [235.73891514 234.20319409 242.28646112]
   [138.65597015  98.97686567  97.26573948]
   [ 64.41166089  50.62760794  49.91755436]
   [195.16242643 161.7629264  141.29947472]
   [213.81001526 198.45441166 192.29735082]
   [223.04552396 216.87677669 224.34137095]]
  ```

#### <u>Nhận xét</u>:

Nhìn chung, chương trình cho ra kết quả khá tốt, có thể chấp nhận được so với hàm `KMeans()` của `scikit-learn`, lý do mà kết quả ra không như mong muốn là càng về lúc sau, **centroid** được cập nhật càng ít đi, do đó số lượng `max_iter` chưa đủ để cho chương trình đưa ra kết quả tốt, ngoài ra, do thư chương trình chỉ sử dụng đơn thuần thư viện `Numpy` nên việc tính toán trên ma trận có kích thước lớn lâu hơn so với việc sử dụng các thư viện khác (vd như `Scipy`).