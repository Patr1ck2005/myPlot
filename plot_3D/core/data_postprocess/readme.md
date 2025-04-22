**函数：`group_surfaces_one_sided_hungarian` 技术文档**  

---

## 1. 简介

在多维参数空间中，经常需要将一组无序的、在每个离散格点上测量或模拟得到的多条曲线（或多层曲面）进行“分组”或“配对”——即在各个格点上，将打乱顺序的样本还原到一致的曲面索引下。  

本算法综合使用基于一阶单边有限差分的导数估计与匈牙利（Hungarian）最优指派算法，通过最小化局部值差和导数不连续性带来的惩罚，实现沿各参数维度的曲面连续分组。

---

## 2. 数学原理

### 2.1 曲面分组建模

- **问题描述**：在维度为 $n$ 的网格上，每个格点 $\mathbf{i}=(i_1,\dots,i_n)$ 处，存在 $m$ 条真实曲面对应的 $m$ 个复值样本 $\{Z_s(\mathbf{i})\}_{s=1}^m$，但我们观测到的是打乱顺序的集合 $\{c_1,\dots,c_m\}$。
- **目标**：为每个格点 $\mathbf{i}$ 恢复索引映射 $\pi_\mathbf{i} : \{1,\dots,m\}\to\{1,\dots,m\}$，使得还原后的曲面在值域与导数上局部连续。

### 2.2 导数估计——一阶单边有限差分

对于已分组好的格点 $\mathbf{j}$（邻点），第 $s$ 条曲面在方向 $d$ 上的一阶导数近似：
$$
\partial_d Z_s(\mathbf{j})
\approx \begin{cases}
\frac{Z_s(j_1,\dots,j_d,\dots) - Z_s(j_1,\dots,j_d-1,\dots)}{\Delta_d}, & \text{后向可用} \\
\frac{Z_s(j_1,\dots,j_d+1,\dots) - Z_s(j_1,\dots,j_d,\dots)}{\Delta_d}, & \text{前向可用} \\
0, & \text{否则}
\end{cases}
$$
其中 $\Delta_d$ 为第 $d$ 维的网格间距。

### 2.3 代价函数构造

对于待分组格点 $\mathbf{i}$，候选值集合 $\{c_1,\dots,c_m\}$，参考格点 $\mathbf{j}$ 第 $s$ 条曲面值 $v_s$ 及各向导数 $d_{s,d}$，我们为匹配 $(s, c_k)$ 定义代价：
$$
C_{s,k} = |c_k - v_s| 
+\sum_{d=1}^n \lambda_d \left|\frac{c_k - v_s}{\Delta_d} - d_{s,d}\right|
$$
- 第一项惩罚值不连续。  
- 第二项惩罚导数不连续，$\{\lambda_d\}$ 为各维度权重。

在 $m\times m$ 的代价矩阵 $C$ 上，使用匈牙利算法求解全局最优一一指派，使得总代价最小。

---

## 3. 算法流程

1. **初始化**  
   - 将输入 `Z` 构造为 `dtype=object` 的 $n$ 维 `np.ndarray`，每个格点存放打乱的 $m$ 个复值。  
   - 创建同 shape 的 `Zg`（object 数组）与 `assigned`（bool 数组）。  
   - 将原点 $\mathbf{0}$ 处的 $m$ 值按实部排序，赋给 `Zg[0,…,0]`，并标记 `assigned[0,…,0]=True`。

2. **遍历所有格点**  
   - 按字典序 (lexicographic) 枚举 $\mathbf{i}\in\prod_{d=1}^n[0,\dots,N_d-1]$；  
   - 若 `assigned[i]` 已 True，跳过；否则：  
     a. 在各维度优先后向邻点中，寻找第一个已赋值的 `neighbor`，记录其对应轴 `axis`；若无，回退到原点。  
     b. 从 `Zg[neighbor]` 取出排序后的复值数组 $v_s$，并用 `estimate_derivative_one_sided` 计算每条曲面在各维度的导数 $d_{s,d}$。  
     c. 构造代价矩阵 $C_{s,k}$ 并调用 `linear_sum_assignment(C)`，获得最优匹配 `(s, k)`。  
     d. 根据匹配顺序，将 `Zg[i][s] = candidates[k]`，并标记 `assigned[i]=True`。  

3. **返回** 完整的 `Zg`，元素为按曲面索引排列的复值数组。

---

## 4. 编程实现细节

- **数据结构**  
  - `Z`: `np.ndarray(shape=dims, dtype=object)`，每个元素是 `list[complex]`。  
  - `Zg`: 同型 `object` 数组，存放 `np.array(complex, shape=(m,))`。  
  - `assigned`: `np.ndarray(shape=dims, dtype=bool)`。  

- **函数接口**  
  ```python
  def group_surfaces_one_sided_hungarian(
      Z: np.ndarray, 
      deltas: Sequence[float],
      lams:   Sequence[float]
  ) -> np.ndarray:
      """
      Z: object 型 n 维数组，
      deltas, lams: 长度 n 的网格步长与权重
      返回同 shape 的 object 型数组 Zg。
      """
  ```

- **导数估计**  
  ```python
  def estimate_derivative_one_sided(
        Zg, assigned, idx: tuple, s: int,
        axis: int, delta: float
    ) -> float:
      # 优先后向、再前向，邻点必须已 assigned
  ```

- **匈牙利匹配**  
  ```python
  C = np.zeros((m,m))
  for s in range(m):
      for k, c in enumerate(candidates):
          C[s,k] = abs(c - v_prev[s])
                  + sum(lams[d]*abs((c-v_prev[s])/deltas[d] - d_prev[s,d])
                        for d in range(n_dims))
  row, col = linear_sum_assignment(C)
  ```

---

## 5. 示例代码

以 3 维 $(nx,ny,nz)=(30,20,15)$，$m=3$ 的示例：

```python
# ... 参见上文完整的 main() 测试 
Zg3 = group_surfaces_one_sided_hungarian(Z3, deltas3, lams3)
```

---

## 6. 复杂度与性能

- **时间复杂度**：
  - 格点遍历 $O(\prod_d N_d)$；
  - 每点构造代价矩阵 $O(m^2 n)$；
  - 匈牙利算法 $O(m^3)$。
  - 总计 $O\bigl(\prod_d N_d\times(m^2 n + m^3)\bigr)$。

- **内存开销**：主要在 `Z, Zg` 两个 object 数组以及 `assigned` 布尔数组。

---

## 7. 限制与扩展

- 仅利用单一邻点方向的导数；可考虑同时利用多方向或更高阶差分。  
- 对噪声敏感：可结合平滑或正则化策略。  
- 对于大 $m$ 或高维度，计算量急剧增加，可尝试稀疏匹配或并行优化。

---

## 8. 结语

本算法在不规则采样或乱序观测中，能够有效保持曲面在值与导数上的局部连续性，为多维数据重建提供了一种灵活、高效的方法。
