import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设输入嵌入向量x的维度是(batch_size, sequence_length, d_model)
# 假设位置编码pos_encoding的维度也是(batch_size, sequence_length, d_model)
# 我们将这两个相加得到最终的输入表示

# 示例参数
batch_size = 64
sequence_length = 100
d_model = 512
num_heads = 8
d_k = d_v = d_model // num_heads

# 示例输入
x = torch.randn(batch_size, sequence_length, d_model)
pos_encoding = torch.randn(batch_size, sequence_length, d_model)

# 合并位置编码和输入嵌入
input_embedding = x + pos_encoding

# 线性变换得到Q, K, V
wq = nn.Linear(d_model, d_k * num_heads, bias=False)
wk = nn.Linear(d_model, d_k * num_heads, bias=False)
wv = nn.Linear(d_model, d_v * num_heads, bias=False)

q = wq(input_embedding).view(batch_size, sequence_length, num_heads, d_k)
k = wk(input_embedding).view(batch_size, sequence_length, num_heads, d_k)
v = wv(input_embedding).view(batch_size, sequence_length, num_heads, d_v)

# 缩放点积注意力
scaled_attention = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
attention_weights = F.softmax(scaled_attention, dim=-1)

# 计算多头注意力的输出
out = torch.matmul(attention_weights, v)
out = out.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

print(out)
print(attention_weights)