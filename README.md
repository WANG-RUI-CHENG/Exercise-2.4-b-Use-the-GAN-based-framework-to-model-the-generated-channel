# Exercise 2.4(b) – Channel GAN Implementation

實作 **Conditional Generative Adversarial Network (CGAN)**，用來學習並模擬 **Rayleigh fading channel** 的接收訊號分布。  
通道資料集來自 **Exercise 2.4(a)** 使用 **QuaDRiGa** 生成的 `rayleigh_channel_dataset.mat`。

本實驗的目標是讓 GAN 在**不直接做顯式通道估測**的情況下，學習無線通道的統計特性，並生成接近真實通道輸出的接收信號。

---

# Project Goal

希望學習以下通道模型：

\[
y = h x + n
\]

其中：

- \(h\)：從 QuaDRiGa 產生的 Rayleigh fading channel coefficient
- \(x\)：隨機產生的 16-QAM symbol
- \(n\)：加性高斯白雜訊 (AWGN)
- \(y\)：接收信號

GAN 的任務是學習條件分布：

\[
p(y \mid x, h)
\]

也就是在給定傳送符號與通道資訊的情況下，生成與真實接收訊號接近的樣本。





---

# Environment Requirements

使用 **Python + TensorFlow v1**。



## Required Packages

先安裝以下套件：

```bash
pip install numpy scipy matplotlib tensorflow==1.15
```


---

# Dataset Preparation

需要先有由 **Exercise 2.4(a)** 生成的通道資料集：

```text
rayleigh_channel_dataset.mat
```

資料內容為：

```text
h_siso : complex channel coefficients
```

程式會用以下方式讀取：

```python
import scipy.io as sio

mat_file_path = 'rayleigh_channel_dataset.mat'
mat_data = sio.loadmat(mat_file_path)
h_dataset = mat_data['h_siso'].flatten()
```

---

# Model Overview

採用 **Conditional GAN (CGAN)**。

## Generator

Generator 的輸入為：

- Noise vector \(z\)
- Conditioning vector

輸出為：

- 生成的接收信號 \([Re(y), Im(y)]\)

---

## Discriminator

Discriminator 的輸入為：

- 真實或生成的接收信號
- Conditioning vector

輸出為：

- 該樣本是否來自真實資料分布

---

# Conditioning Vector

conditioning vector 定義為：

\[
[Re(x), Im(x), Re(h), Im(h)]
\]

也就是將：

- transmitted symbol 的實部與虛部
- channel coefficient 的實部與虛部

串接起來，並做簡單正規化：

```python
conditioning = np.hstack((
    np.real(data).reshape(number, 1),
    np.imag(data).reshape(number, 1),
    h_r.reshape(number, 1),
    h_i.reshape(number, 1)
)) / 3.0
```

---

# Real Sample Generation

訓練資料的生成方式如下：

1. 從 `h_dataset` 隨機選取 channel coefficient \(h\)
2. 隨機產生 16-QAM symbol \(x\)
3. 根據通道模型產生接收訊號

\[
y = hx + n
\]

4. 將接收訊號拆成實部與虛部
5. 建立 conditioning vector

---

# Core Function

以下是 Exercise 2.4(b) 中最重要的資料生成函式：

```python
def generate_real_samples_with_labels_Rayleigh(h_dataset, number=100):
    # randomly select channel coefficients h
    h_complex = np.random.choice(h_dataset, number)
    h_r = np.real(h_complex)
    h_i = np.imag(h_complex)

    # randomly generate QAM symbols x
    labels_index = np.random.choice(len(mean_set_QAM), number)
    data = mean_set_QAM[labels_index]

    # received signal y = h*x + n
    received_complex = h_complex * data
    received_data = np.hstack(
        (
            np.real(received_complex).reshape(number, 1),
            np.imag(received_complex).reshape(number, 1)
        )
    )

    gaussian_random = np.random.multivariate_normal(
        [0, 0], [[0.01, 0], [0, 0.01]], number
    ).astype(np.float32)

    received_data = received_data + gaussian_random

    # conditioning vector = [Re(x), Im(x), Re(h), Im(h)] / 3
    conditioning = np.hstack(
        (
            np.real(data).reshape(number, 1),
            np.imag(data).reshape(number, 1),
            h_r.reshape(number, 1),
            h_i.reshape(number, 1)
        )
    ) / 3.0

    return received_data.astype(np.float32), conditioning.astype(np.float32)
```

---

# Training Configuration

本專案使用以下訓練設定：

| Parameter | Value |
|----------|------|
| Batch Size | 512 |
| Condition Dimension | 4 |
| Noise Dimension \(Z\) | 16 |
| Training Data Size | 10000 |
| Training Iterations | 750000 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| GAN Type | WGAN-GP |

---

# QAM Constellation

本實驗使用 **16-QAM** constellation：

```python
mean_set_QAM = np.asarray([
    -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
    -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
     1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
     3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j
], dtype=np.complex64)
```

---

# How to Run

確認目前資料夾內有：

```text
Exercise_2.4_starter.py
rayleigh_channel_dataset.mat
```

資料來源由Exercise_2.4(a) 產出

---

# Training Output

訓練過程中會產生以下輸出：

## 1. Model Checkpoints

儲存在：

```text
Models/
```

---

## 2. Generated Images

儲存在：

```text
ChannelGAN_Rayleigh_images/
```

這些圖片用來比較：

- 真實接收信號
- GAN 生成的接收信號

若訓練成功，GAN 生成的分布會逐漸逼近真實資料分布。

---

# Expected Result

## Exercise 2.4(b) GAN-based Channel Modeling Result

使用條件式生成對抗網路（Conditional GAN, CGAN）來學習由 QuaDRiGa 所生成之通道資料分布，目標是在未知通道模型的情況下，透過資料驅動方式逼近通道輸出 ( y = hx + n ) 的統計特性。實驗中以 Rayleigh fading channel 為例，並結合 QAM 調變訊號作為輸入條件，訓練生成器以產生對應之接收訊號分布。

在訓練初期（step 0），生成器尚未學習到真實通道的分布特性，生成樣本呈現嚴重的集中現象（mode collapse），僅分布於狹小區域，與真實資料（real samples）之分散型態存在顯著差異，顯示此時模型尚未建立輸入與通道之間的對應關係。

隨著訓練進行，生成器逐漸學習到通道對不同 QAM symbol 的影響，生成樣本開始形成多個對應於星座點的群集（clusters），並圍繞在理論接收點 ( hx ) 附近，同時呈現合理的雜訊擴散效果。此結果顯示 CGAN 已成功捕捉通道的統計特性與條件分布。

從結果對比圖可以觀察到，訓練完成後的生成樣本分布與真實資料高度相似，不僅在整體形狀上吻合，也能反映不同符號對應之接收訊號結構，證明 GAN-based framework 能有效學習並重建通道行為。

綜合而言，本實驗驗證了 GAN 在無需明確通道模型的情況下，仍可透過資料學習方式準確逼近無線通道分布，對於未來通訊系統中資料驅動通道建模具有潛在應用價值。


成功訓練後，GAN 應該能夠生成與真實 Rayleigh 通道接收訊號相近的複數分布。  
在 constellation 圖中，生成樣本的分布應逐漸接近：

\[
y = hx + n
\]

所形成的真實接收信號點雲。

<img width="2400" height="1000" alt="image" src="https://github.com/user-attachments/assets/e4bcd33f-10e8-4bda-af76-53d308b513ff" />


---

# Ideal Comparison

此圖比較 GAN 生成樣本與理論接收點 ( hx )。生成樣本分布於理想點周圍，顯示模型已捕捉通道映射關係，而分散程度反映通道雜訊影響。

<img width="511" height="508" alt="image" src="https://github.com/user-attachments/assets/18f992c8-2c6b-416d-9e58-0c50f810d4e9" />

---

# Discussion

從結果可觀察到：

訓練初期存在 mode collapse 問題

經過訓練後，GAN 能有效學習通道分布

模型成功捕捉 symbol 與 channel 之間的對應關係

生成結果與真實通道輸出高度一致

---

# Notes

1. 本程式使用 `tensorflow.compat.v1`，因此建議使用 **TensorFlow 1.15**。
2. 若沒有 GPU，可將以下兩行註解掉或改成 CPU 設定：

```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
```

3. 若 `Models/` 資料夾不存在，請先建立，或在程式中加入：

```python
if not os.path.exists('./Models'):
    os.makedirs('./Models')
```
