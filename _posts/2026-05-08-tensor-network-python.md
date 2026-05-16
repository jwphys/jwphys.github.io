---
layout: post
title: "Python 中的张量网络计算"
date: 2026-05-08
tags: [python, tensor-network, numerical]
lang: zh
---

张量网络是量子多体物理中的重要工具。

## 什么是张量网络？

张量网络是一种高效表示高维张量的方法。它将一个巨大的张量分解为多个小张量的乘积网络。

## 常见类型

- **MPS** (Matrix Product States) — 一维系统
- **PEPS** (Projected Entangled Pair States) — 二维系统
- **MERA** (Multi-scale Entanglement Renormalization Ansatz) — 临界系统

## Python 库

```python
import numpy as np
import tensornetwork as tn

# 创建两个节点
a = tn.Node(np.random.rand(2, 2))
b = tn.Node(np.random.rand(2, 2))

# 连接边
edge = a[1] ^ b[0]
result = tn.contract(edge)
```

## 参考

- Schollwöck, U. (2011). *The density-matrix renormalization group in the age of matrix product states*. Annals of Physics, 326(1), 96-192.
