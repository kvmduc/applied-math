# TOÁN ỨNG DỤNG VÀ THỐNG KÊ
Course `MTH00051`: Applied Mathematics and Statistic
Class 18CLC4 Term III / 2019-2020 

Author:
-   `18127080`: Kiều Vũ Minh Đức (Github: [@kvmduc](https://github.com/kvmduc))

---

# PROJECT 3 : Least Square Error - Linear Regression

## Ý tưởng thực hiện:

Trong đồ án này có nhiều yêu cầu thực hiện xử lý khác nhau , chúng ta sẽ phân tích ý tưởng riêng ở từng yêu cầu

Các thư viện sử dụng : 

+ `numpy` sử dụng để tính toán chính trong đồ án này
+ `pandas` sử dụng để đọc file csv và biến dữ liệu thành dataframe
+ `tensorflow` sử dụng để tạo mô hình autoencoder cơ bản
+ `sklearn` sử dụng để phân chia dataset dựa vào thuật toán k-fold cross validation
+ `operator` sử dụng để có những hàm thao tác trên list mở rộng hơn

### Giới thiệu các bước thực hiện của thuật toán

Nghiệm của phương trình hồi quy tuyến tính $Ax = b$ được tính bằng $\hat{x} = A^\dagger b$

và Loss của mô hình so với kết quả được tính bằng $\hat{r} = \lVert A \hat{x} - b \rVert^2$

Em sẽ tính dựa vào norm-2 để có được kết quả loss của mô hình.

Ngoài ra em cũng có sử dụng một phương pháp toán có tên là **bias trick**, ý tưởng này tương đối đơn giản, nhưng có thể làm tăng độ chính xác của mô hình lên tương đối nhiều. Ý tương là thay vì sử dụng mô hình $Ax = b$ thì ta có thể cộng thêm một giá trị bias để mô hình không bị ràng buộc phải qua gốc tọa độ :
$$
Ax + bias = b
$$
Nếu xem $A_0 = 1$ thì có thể thể biến phương trình trên về dạng :
$$
Ax + bias * A_0= b
$$
 Ta có thể xem giá trị bias là một giá trị có thể học được từ dữ liệu.

Trước tiên ta quy chuẩn dữ liệu để phân ra test và validation theo tỉ lệ (9:1), em sử dụng hàm `sklearn.model_selection.train_test_split()`

#### <u>Yêu cầu 1</u> : Sử dụng toàn bộ 11 đặc trưng đề bài cung cấp :

Sau khi phân data ra làm 2 cụm là train và validation, chúng ta sẽ đảm bảo hơn việc đánh giá chính xác hơn khi cho dư liệu để kiểm tra không thuộc tập train. Áp dụng phương pháp bias trick trên, ta  tìm được mô hình của dữ liệu trên :

```
[[ 5.80908475e+01]
 [ 6.90090074e-02]
 [-9.77935466e-01]
 [-2.23273740e-01]
 [ 3.84683433e-02]
 [-1.59781533e+00]
 [ 5.91568620e-03]
 [-3.96178504e-03]
 [-5.52885030e+01]
 [-1.36227495e-01]
 [ 8.91465906e-01]
 [ 2.58230133e-01]]
```

Với phần tử đầu tiên của mô hình là phần tử bias.

Loss của mô hình, với cách tính như trên, có giá trị (Loss được tính dựa trên tập validation, sai số của toàn bộ tập validation):

```
61.9158
```

Lý giải cho loss cao là vì bài dữ liệu của bài toán này thuộc về bài toán **classification**, kết quả sau khi được mô hình trả về có thể là số thập phân (Ví dụ như kết quả của mô hình có thể trả ra 5.1 trong khi label lại có giá trị là 5)

#### <u>Yêu cầu 2</u> : Sử dụng toàn bộ 1 đặc trưng và so sánh các đặc trưng với nhau :

Sau khi sử dụng hàm `sklearn.model_selection.KFold()` để chia cross validation, với số lượng k = 10 

Tương tự như trên, ta áp dụng thuật toán đối với từng feature một, ta sẽ có lần lượt kết quả như sau:

|                          |       fixed acidity       |      volatile acidity      | citric acid               | residual sugar            | chlorides                  | free sulfur dioxide       | total sulfur dioxide                | density                      | pH                         | sulphates                 | alcohol                   |
| :----------------------: | :-----------------------: | :------------------------: | ------------------------- | ------------------------- | -------------------------- | ------------------------- | ----------------------------------- | ---------------------------- | -------------------------- | ------------------------- | ------------------------- |
| Weight(weight[0] = bias) | [5.22060945] [0.05560935] | [6.58965931] [-1.71585346] | [5.41722195] [0.96056164] | [5.62718677] [0.03057864] | [5.91013543] [-2.32054217] | [5.80214709] [-0.0063143] | [ 5.95359595e+00] [-5.41418251e-03] | [95.35259807] [-89.9080009 ] | [6.88908482] [-0.35889662] | [4.78973962] [1.38530316] | [1.77191849] [0.37641206] |
|           Loss           |          62.9597          |          59.5077           | 66.6851                   | 69.0794                   | 68.7493                    | 69.4857                   | 63.7803                             | 73.8487                      | 68.2353                    | 93.1135                   | 54.8576                   |

Dựa vào kết quả trên, ta có thể rút ra được mô hình dựa trên thuộc tính **alcohol** có Loss value thấp nhất, nên đây là thuộc tính đáng tin cậy nhất.

#### <u>Yêu cầu 3</u> : Xây dựng một mô hình của riêng bạn cho kết quả tốt nhất :

Ở yêu cầu trên, do mo hình Linear Regression nhạy cảm với noise trong data, ví dụ :

![img](https://machinelearningcoban.com/assets/LR/output_13_1.png)

Nên em nghĩ ra một hướng đó là có thể sử dụng một mô hình **autoencoder** đơn giản, qua đó có thể lọc bớt noise trong data set, ngoài ra ở kết quả chạy thực nghiệm ở 2 yêu cầu trên, ta có thể so sánh Loss value nếu chúng ta sử dụng 11 feature sẽ có khả năng cao hơn nếu chúng ta sử dụng ít feature (điển hình nếu sử dụng  feature alcohol cho việc dựng mô hình). Nên sử dụng autoencoder để có thể qua đó giảm số feature để dựng mô hình.

Đây là kiến trúc của mô hình autoencoder:

![Basics of Autoencoders. Autoencoders (AE) are type of… | by Deepak Birla |  Medium](https://miro.medium.com/max/600/1*nqzWupxC60iAH2dYrFT78Q.png)

Chúng ta sẽ xây dụng và train model này sao cho input đầu vào và output đầu ra có sự sai khác nhỏ nhất có thể. sau đó lấy output đầu ra ở đoạn **bottle neck** để có được số feature giảm đi so với 11 feature ban đầu.

Ở trong code, em có xây dụng một mô hình autoencoder đơn giản với 3 layer dense, mỗi layer sẽ có **activation function** là $ReLU$ do các input đầu vào là giá trị của các feature không bị chặn trên và chặn dưới như $Sigmoid$ hay $Hyperbolic-Tangent$ . Sau đó thực hiện việc train model trên với dữ liệu là một vector có 11 feature sao cho đầu giá trị đầu vào và giá trị đầu ra có sai số nhỏ nhất.

Sau khi train thành công. Em lấy output với **output_size = 6**, để thực hiện việc xây model. Kết quả trả ra ở mỗi lần chạy trung bình cho kết quả tốt hơn một số feature

+ Lần chạy 1:

Loss:

```
58.08916173814126
```

Mô hình có kết quả :

```
[[ 1.57365621]
 [ 0.23204614]
 [ 0.21761653]
 [-0.00737211]
 [ 0.04975997]
 [-0.13902395]
 [ 0.        ]]
```

+ Lần chạy 2:

Loss:

```
58.088945337313554
```

Mô hình có kết quả :

```
[[ 1.44949337e+00]
 [-1.05535938e-14]
 [ 3.54368982e-01]
 [-7.85331421e-02]
 [-5.64911671e-04]
 [ 4.09483828e-02]
 [-9.41162040e-02]]
```

+ Lần chạy 3:

Loss:

```
75.15554007005599
```

Mô hình có kết quả :

```
[[ 3.92762774e+00]
 [ 1.16054344e-01]
 [-2.05815780e-17]
 [-2.56112745e-02]
 [ 1.08592038e-01]
 [ 0.00000000e+00]
 [ 0.00000000e+00]]
```



Có một số lần chạy do thuộc tính chọn ra được encode có giá trị tương đối nhỏ, nên sẽ trả ra bằng 0, feature vô nghĩa, do đó kết quả dựng mô hình không tốt. Ngoài ra qua **yêu cầu 2** ta có thể thấy được một số feature có Loss tương đối cao (điển hình như feature  **sulphates** có $Loss \approx 93$ )

+ Lần chạy 4:

Loss:

``` 
58.23132342934443
```

Mô hình có kết quả :

```
[[ 1.97266596]
 [-0.06730383]
 [-0.00433466]
 [-0.11638817]
 [ 0.00292563]
 [ 0.        ]
 [ 0.34896697]]
```

+ Lần chạy 5:

Loss:

```
57.122222517128776
```

Mô hình có kết quả:

```
[[ 1.27323891e+00]
 [ 4.59418289e-02]
 [ 6.25950559e-15]
 [-1.37978349e-01]
 [-1.33648321e-02]
 [ 1.30644182e-01]
 [ 1.91917001e-01]]
```

Ta có thể thấy ở lần chạy này, sau khi encode xong vẫn còn giữ nhiều thông tin, do đó kết quả trả về tương đối tốt

### Citation 

Đồ án này em có tham khảo một số tài liệu như :

[Machine Learning Cơ bản](https://machinelearningcoban.com/2016/12/28/linearregression/)

[Keras](https://keras.io/api/)

[towardsdatascience](https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726)

