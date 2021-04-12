TOÁN ỨNG DỤNG VÀ THỐNG KÊ

Course `MTH00051`: Applied Mathematics and Statistic
Class 18CLC4 Term III / 2019-2020 

Author:
-   `18127080`: Kiều Vũ Minh Đức (Github: [@kvmduc](https://github.com/kvmduc))

---

# PROJECT 2 : Image Processing 

## Ý tưởng thực hiện:

Trong đồ án này có nhiều yêu cầu thực hiện xử lý ảnh khác nhau (Image processing), chúng ta sẽ phân tích ý tưởng riêng ở từng yêu cầu

### Giới thiệu các bước thực hiện của thuật toán

#### <u>Yêu cầu 1</u> : Tăng độ sáng cho ảnh & <u>Yêu cầu 2</u> : Tăng độ tương phản cho ảnh

Theo thuật toán được đề xuất trong cuốn [Digital Image Processing](https://www.amazon.com/Digital-Image-Processing-Rafael-Gonzalez/dp/0133356728) của tác giả Rafael C. Gonzalez, ông có định nghĩa việc thay độ sáng (<b>brightness</b>) và độ tương phản (<b>contrast</b>) bằng công thức toán sau
$$
g(i,j) = \alpha f(i,j) + \beta
$$
Với $\alpha$ là hệ số độ tương phản mà ta muốn thay đổi (mặc định của một bức ảnh khi đọc vào, $\alpha = 1$) và $\beta$ là hệ số độ sáng mà ta muốn tăng ($\beta$ có thể nhỏ hơn 0 để giảm độ sáng của bức ảnh). Ta có thể hiểu ý nghĩa như sau:

+ Do một điểm ảnh $f(i,j)$ sẽ mang giá trị trong đoạn $[0;255]$ với $f(i,j) = 0$ thì điểm ảnh đó sẽ thể hiện màu đen hoàn toàn và $f(i,j) = 255$ thì điểm ảnh sẽ thể hiện một màu trắng hoàn toàn. Nói cách khác, nếu giá trị càng cao, màu điểm ảnh sẽ càng tiến về màu sáng, định nghĩa trên cũng đúng với ảnh RGB do ảnh RGB được ghép lại bởi 3 kênh màu Red, Green, Blue. Nếu ta cộng giá trị này cho một giá trị $\beta$ thì các màu sẽ cùng đẩy lên một giá trị, với giá trị $\beta > 0$, các màu sẽ cùng sáng hơn. Đó là ý nghĩa về việc tăng độ sáng của ảnh :
  + Để xử lý, chúng ta cần một số hàm ```numpy``` , do `ndarray` được hỗ trợ có thể broadcasting, nên chúng ta chỉ cần cộng ma trận ảnh cho một **bias** thì `numpy` đã có thể tự hiểu là cộng toàn bộ các giá trị cho **bias** đó. Sau đó sử dụng hàm `numpy.clip()` để có thể giới hạn giá trị đầu ra của phép cộng. Các giá trị phải thuộc đoạn $[0;255]$
  + Để dễ dàng hơn cho việc xử lý, do ảnh màu có nhiều hơn 2 chiều (có thể là 3 đối với ảnh RGB, 4 đối với RGB-$\alpha$ ) do đó có thể flatten ảnh ra 2 chiều, sau khi xử lý xong có thể reshape lại như ban đầu. Dưới đây là hàm để thay đổi độ sáng.

```python
def change_brightness(img_1d, bias):
    row, column, channel = img_1d.shape
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    img_1d=  img_1d.astype(np.uint16) + bias
    img_1d = np.clip(img_1d,0,255)
    img_1d = img_1d.reshape((row, column, channel))
    return img_1d	
```

+ Đối với độ tương phản, khi càng tăng độ tương phản, tức là càng tăng khoảng cách giá trị giữa các màu. Một cách đơn giản đó là có thể nhân toàn bộ ảnh cho một giá trị $\alpha > 0$ để làm tăng khoảng cách giá trị màu giữa các điểm ảnh. Nếu giá trị $\alpha > 1$, ta sẽ có ảnh đã được tăng độ tương phản, hãy tưởng tượng chúng ta đang có $a = 1 , b = 3$ khoảng cách hiện tại giữa chúng đang là $\delta = 2$, nếu ta cùng nhân cả hai giá trị $a,b$ cho 2, khoảng cách giữa chúng lúc sau sẽ tăng ($\delta = 4$). Ngược lại đối với giá trị $0 < \alpha < 1$
  + Tương tự như trên, giá trị của các kênh màu phải thuộc đoạn $[0;255]$, nên ta cũng cần sử dụng hàm `numpy.clip()`, tận dụng khả năng broadcasting, ta có thể viết ngắn gọn hàm tăng độ tương phản như sau. 

```python
def change_contrast(img_1d, alpha):
    row, column, channel = img_1d.shape
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    img_1d=  img_1d.astype(np.uint16) * alpha
    img_1d = np.clip(img_1d,0,255)
    img_1d = img_1d.reshape((row, column, channel))
    return img_1d
```

##### Kết quả chạy thử :

<u>Sample</u> :

![img_1](D:\Python Project\Applied Math\Lab02\img_1.jpg)

Sau khi chạy hàm `change_brightness(image,bias)` với $\beta = bias = 100$, ta được :

![brighter](D:\Python Project\Applied Math\Lab02\brighter.png)

Sau khi chạy hàm `change_brightness(image,bias)` với $\beta = bias = -100$, ta được :

![darker](D:\Python Project\Applied Math\Lab02\darker.png)

Sau khi chạy hàm `change_contrast(image,alpha)` với $\alpha = 5$, ta được:

<img src="D:\Python Project\Applied Math\Lab02\higher.png" alt="higher" style="zoom:80%;" />

+ Có thể thấy khác nhau, với `change_brightness()` là mặc dù các màu được tăng độ sáng lên cùng một giá trị, còn ở hàm `change_contrast()` thì sẽ hiện ra một số vùng trắng hoàn toàn, còn một số vùng màu được tăng lên không nhiều.

Sau khi chạy hàm `change_contrast(image,alpha)` với $\alpha = 0.4$, ta được:

![lower](D:\Python Project\Applied Math\Lab02\lower.png)

+ Có thể thấy khác nhau, với `change_brightness()` các màu tối đi, nhưng vẫn thể hiện được độ tươi của từng màu , còn ở hàm `change_contrast()` thì các vùng màu có xu hướng không còn được nổi bật.

#### <u>Yêu cầu 3</u> : Chuyển ảnh sang ảnh xám:

Ảnh xám (grayscale) là ảnh chỉ mang thông tin về độ sáng và tối (độ trắng và đen) của điểm ảnh. Em có hai ý tưởng chuyển từ một ảnh màu sang ảnh grayscale như sau:

+ <u>Average</u>: Cách này sử dụng theo đúng định nghĩa, giá trị của điểm ảnh $f(i,j) = \frac{1}{3}(Red(i,j) + Green(i,j) + Blue(i,j))$, ta sẽ tách từng kênh màu (hoặc giá trị từng kênh màu tại điểm ảnh đang xét), tính trung bình của giá trị ba kênh màu đó, ta sẽ được một giá trị mới. Tuy nhiên ảnh vừa mới tạo có một kênh, ta phải chồng lên ảnh vừa mới tạo hai lần tạo thành một cái tensor tương tự như mô hình ảnh RGB.
  + Ta có thể dễ dàng tính toán giá trị trung bình dựa vào hàm `numpy.mean()`
  + Ngoài ra có thể sử dụng hàm `numpy.concatenate()` hoặc hàm `numpy.dstack()` để hỗ trợ việc chồng các array lên lẫn nhau.

+ Weight: Cách thứ hai sử dụng weight, trong cuốn sách Digital Image Processing có đề xuất giá trị trọng số để chuyển ảnh sang ảnh xám như sau :
  $$
  Y_0(x,y) = 0.299 I_0(x,y) + 0.587 I_1(x,y) + 0.114 I_2(x,y)
  $$
  Với $Y_0(x,y)$ là giá trị mới của điểm ảnh $(x,y)$, $I_0(x,y), I_1(x,y), I_3(x,y)$ lần lượt là giá trị kênh màu Red, Green, Blue tại điểm ảnh $(x,y)$. Các giá trị weight này xuất phát từ ITU-R Recommendation BT. 709.

  ```python
  def change_grayscale(img_1d, way = 'weight'):
      row, column, channel = img_1d.shape
      img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
      if way == 'average':
          new_img_1d = np.zeros((img_1d.shape[0], 1))
          for i in range (new_img_1d.shape[0]):
              new_img_1d[i]+= np.mean(img_1d[i])
          new_img_1d = new_img_1d.reshape((row, column, 1))
          new_img_1d = np.concatenate((new_img_1d, new_img_1d, new_img_1d), -1)
          return new_img_1d
      if way == 'weight':
          r, g, b = img_1d[:, 0], img_1d[:, 1], img_1d[:, 2]
          new_img_1d = 0.2989 * r + 0.5870 * g + 0.1140 * b
          new_img_1d = new_img_1d.reshape(row, column, 1)
          new_img_1d = np.dstack((new_img_1d,new_img_1d,new_img_1d))
          return new_img_1d
  ```

  

##### Kết quả chạy thử :

<u>Sample</u> :

![img_1](D:\Python Project\Applied Math\Lab02\img_1.jpg)

Sau khi chạy hàm `change_grayscale(image,way = 'average')` , ta được :

![average](D:\Python Project\Applied Math\Lab02\average.png)

Sau khi chạy hàm `change_grayscale(image,way = 'weight')` , ta được :

![weight](D:\Python Project\Applied Math\Lab02\weight.png)

#### <u>Yêu cầu 4</u> : Lấy đối xứng ảnh theo trục thẳng đứng:

Để flip ảnh, chúng ta chỉ cần lấy slice ngược ma trận ảnh

![Flip Horizontal](https://help.optitex.com/Flip_Image.png)

+ Đối với horizontal flip, chúng ta sẽ giữ nguyên hàng, lấy ngược từng cột

+ Đối với vertical flip, chúng ta sẽ giữ nguyên cột, lấy ngược từng hàng

```python
def change_reflection(img_1d, way):
    if way == 'horizontal':
        new_img_1d = img_1d[0][::-1]
        for i in range(1,img_1d.shape[0]):
            temp = img_1d[i][::-1]
            new_img_1d = np.concatenate((new_img_1d, temp), axis=0)
        new_img_1d = new_img_1d.reshape((img_1d.shape[0], img_1d.shape[1], img_1d.shape[2]))
        return new_img_1d
    if way == 'vertical':
        new_img_1d = img_1d[::-1][0]
        for i in range(1, img_1d.shape[1]):
            temp = img_1d[::-1][i]
            new_img_1d = np.concatenate((new_img_1d, temp), axis=0)
        new_img_1d = new_img_1d.reshape((img_1d.shape[0], img_1d.shape[1], img_1d.shape[2]))
        return new_img_1d
```

##### Kết quả chạy thử :

<u>Sample</u> :

![img_1](D:\Python Project\Applied Math\Lab02\img_1.jpg)

Sau khi chạy hàm `change_reflection(image,way = 'horizontal')` , ta được :

![horizontal](D:\Python Project\Applied Math\Lab02\horizontal.png)

Sau khi chạy hàm `change_reflection(image,way = 'vertical')` , ta được :

![vertical](D:\Python Project\Applied Math\Lab02\vertical.png)

#### <u>Yêu cầu 5</u> : Chồng ảnh cùng kích thước:

Thuật toán chồng ảnh của openCV đã đề xuất
$$
g(x) = \alpha f_0(x) + (1-\alpha) f_1(x)
$$

+ Do yêu cầu chỉ áp dụng trên ảnh xám, nên trước khi chồng ảnh, ta cần chuyển ảnh về grayscale, sau đó áp dụng công thức trên. Nếu để ý, công thức trên chính là thay đổi độ tương phản của cả hai ảnh sao cho tổng của hệ số nhân với hai ảnh bằng 1. Sau đó cộng lại với nhau

```python
def change_concatenate(img_1d_1, img_1d_2, alpha1):
    img_1d_1 = change_grayscale(img_1d_1)
    img_1d_2 = change_grayscale(img_1d_2)
    new_img_1d = alpha1 * img_1d_1 + (1-alpha1) * img_1d_2
    return new_img_1d
```

##### Kết quả chạy thử :

<u>Sample</u> 1:

![img_1](D:\Python Project\Applied Math\Lab02\img_1.jpg)

<u>Sample</u> 2:

![horizontal](D:\Python Project\Applied Math\Lab02\horizontal.png)

Sau khi chạy hàm `change_reflection(image1, image2, 0.5)`với giá trị $\alpha = 0.5$ , ta được :

![blend](D:\Python Project\Applied Math\Lab02\blend.png)

#### <u>Yêu cầu 6</u> : Làm mờ ảnh:

Để làm mờ ảnh, em áp dụng ý tưởng như trong convolutional-layer trong convolutional neuron network (CNN), ý tưởng đó như sau:

+ Tạo một kernel với kích thước được cho, kích thước phải là số lẻ, do điểm ảnh đang xét phải là trung tâm của kernel, kernel càng lớn, thì ảnh càng mờ do càng bao được nhiều điểm lân cận. Tuy nhiên khác với việc trích xuất như bên dưới, thuật toán của em giữ nguyên kích thước của ảnh, có nghĩa là kích thước đầu ra của Image bằng với kích thước đầu vào bằng cách chỉ lấy kernel của các điểm có thể lấy, nhưng vẫn xét toàn bộ ảnh.

![img](https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/03/giphy.gif?resize=474%2C345&ssl=1)

```python
def change_blur(img_2d, kernel_size = 3):
    if kernel_size % 2 == 0:
        print('Kernel size must be odd \n')
        return img_2d
    padding = (kernel_size - 1 )/ 2
    padding = int(padding)
    new_img_2d = np.zeros((img_2d.shape[0], img_2d.shape[1], img_2d.shape[2]))
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            kernel = img_2d[np.clip(i-padding,0,img_2d.shape[0]):np.clip(i+padding + 1,0,img_2d.shape[0]),np.clip(j-padding,0,img_2d.shape[1]):np.clip(j+padding + 1,0,img_2d.shape[1]),:]
            new_img_2d[i][j][0] = np.mean(kernel[:, :, 0])
            new_img_2d[i][j][1] = np.mean(kernel[:, :, 1])
            new_img_2d[i][j][2] = np.mean(kernel[:, :, 2])
    return new_img_2d
```

##### Kết quả chạy thử :

<u>Sample</u> :

![img](D:\Python Project\Applied Math\Lab02\img.jpg)

Sau khi chạy hàm `change_blur(image,kernel_size = 3)` , ta được :

![blur3](D:\Python Project\Applied Math\Lab02\blur3.png)

Sau khi chạy hàm `change_blur(image,kernel_size = 15)` , ta được :

![blur15](D:\Python Project\Applied Math\Lab02\blur15.png)