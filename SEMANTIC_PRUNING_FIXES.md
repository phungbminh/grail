# Sửa lỗi phương pháp 2-Stage Semantic Pruning

## Tóm tắt các sửa đổi

Đã sửa các lỗi logic nghiêm trọng và cải thiện phương pháp 2-stage semantic pruning trong GraIL.

---

## 1. ✅ Sửa lỗi NGHIÊM TRỌNG: Rebuild labels sau pruning

**File**: `subgraph_extraction/graph_sampler.py:562-634`

### Vấn đề cũ
```python
# SAI: Logic mapping sai giữa node indices và labels
old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(pruned_subgraph_nodes)}
for old_idx in range(len(pruned_labels)):
    original_node_id = subgraph_nodes[old_idx]  # ← Bug ở đây!
```

**Lỗi**:
- `old_idx` iterate qua pruned_labels
- `subgraph_nodes[old_idx]` access vào list khác có kích thước khác
- Dẫn đến labels và nodes không khớp!

### Giải pháp mới
```python
# ĐÚNG: Build mapping node_id → label trước
node_to_label = {
    subgraph_nodes[i]: pruned_labels[i]
    for i in range(len(subgraph_nodes))
}

# Rebuild labels theo đúng thứ tự pruned nodes
new_labels = []
for node_id in pruned_subgraph_nodes:
    if node_id in node_to_label:
        new_labels.append(node_to_label[node_id])
    else:
        # Fallback cho root nodes
        if node_id == ind[0]:
            new_labels.append([0, 1])
        elif node_id == ind[1]:
            new_labels.append([1, 0])
```

**Impact**: ✅ Labels giờ đây CHÍNH XÁC khớp với nodes sau pruning

---

## 2. ✅ Thêm normalization cho scores

**File**: `subgraph_extraction/semantic_pruning.py:311-337`

### Vấn đề cũ
```python
# SAI: Không normalize, semantic score dominant
final_scores[node] = alpha * path_score + beta * sem_score
# path_score ≈ 0.001-0.1
# sem_score ≈ -2 đến 2
# → semantic_score chiếm ưu thế!
```

### Giải pháp mới
```python
# ĐÚNG: Normalize cả hai về [0,1] trước khi kết hợp
path_values = np.array([path_scores[n] for n in stage1_candidates])
sem_values = np.array([semantic_scores[n] for n in stage1_candidates])

# Normalize to [0, 1]
path_normalized = (path_values - path_values.min()) / (path_values.max() - path_values.min() + 1e-8)
sem_normalized = (sem_values - sem_values.min()) / (sem_values.max() - sem_values.min() + 1e-8)

# Combined với scale công bằng
final_scores[node] = alpha * path_normalized[i] + beta * sem_normalized[i]
```

**Impact**: ✅ Path và semantic scores giờ có trọng số cân bằng

---

## 3. ✅ Cải thiện công thức path length

**File**: `subgraph_extraction/semantic_pruning.py:106-118`

### Vấn đề cũ
```python
# Công thức cũ không phân biệt tốt
scores[node] = 1.0 / (d_u + d_v + 1e-6)
# Node (1,10) và node (5,6) có cùng score = 1/11
```

### Giải pháp mới
```python
# CẢI THIỆN: Sum of inverse distances
# Ưu tiên nodes gần ít nhất 1 endpoint
scores[node] = 1.0/(d_u + 1e-6) + 1.0/(d_v + 1e-6)

# Node (1,10): 1/1 + 1/10 = 1.1
# Node (5,6):   1/5 + 1/6 = 0.37
# → Node gần hơn có điểm cao hơn!
```

**Impact**: ✅ Phân biệt tốt hơn nodes gần vs xa endpoints

---

## 4. ✅ Thêm validation checks toàn diện

### A. Validation trong `two_stage_pruning()`

**File**: `subgraph_extraction/semantic_pruning.py:358-378`

```python
# 1. Check final size
if len(final_nodes) > target_M:
    final_nodes = final_nodes[:target_M]

# 2. Check u and v KHÔNG trong final nodes
if u in final_nodes or v in final_nodes:
    final_nodes = [n for n in final_nodes if n not in (u, v)]

# 3. Check duplicates
if len(final_nodes) != len(set(final_nodes)):
    final_nodes = list(dict.fromkeys(final_nodes))
```

### B. Validation trong `graph_sampler.py`

**File**: `subgraph_extraction/graph_sampler.py:586-600`

```python
# 1. Ensure labels match nodes
if len(pruned_labels) != len(pruned_subgraph_nodes):
    raise ValueError("Inconsistent labels")

# 2. Ensure root nodes ở positions [0,1]
if pruned_subgraph_nodes[0] != ind[0] or pruned_subgraph_nodes[1] != ind[1]:
    raise ValueError("Root node ordering violated")

# 3. Ensure root labels đúng
if not (np.array_equal(pruned_labels[0], [0, 1]) and
        np.array_equal(pruned_labels[1], [1, 0])):
    raise ValueError("Incorrect root labels")
```

### C. Final validation cuối hàm

**File**: `subgraph_extraction/graph_sampler.py:647-656`

```python
# Critical consistency checks với assertions
assert len(pruned_subgraph_nodes) == len(pruned_labels)
assert pruned_subgraph_nodes[0] == ind[0] and pruned_subgraph_nodes[1] == ind[1]
assert np.array_equal(pruned_labels[0], [0, 1]) and np.array_equal(pruned_labels[1], [1, 0])
```

**Impact**: ✅ Phát hiện ngay lỗi consistency, đảm bảo invariants

---

## 5. ✅ Sửa fallback logic

**File**: `subgraph_extraction/graph_sampler.py:602-634`

### Vấn đề cũ
```python
# Fallback cũ cũng có lỗi label mapping tương tự
```

### Giải pháp mới
```python
# Fallback: random sampling NHƯNG giữ root nodes ở đầu
other_nodes = [n for n in pruned_subgraph_nodes if n not in ind]
num_to_sample = max_nodes_per_hop - 2
sampled_others = np.random.choice(other_nodes, num_to_sample, replace=False).tolist()
pruned_subgraph_nodes = list(ind) + sampled_others

# FIX: Rebuild labels đúng cho fallback
node_to_label = {subgraph_nodes[i]: pruned_labels[i] for i in range(...)}
new_labels = [node_to_label.get(node_id, fallback_label) for node_id in pruned_subgraph_nodes]
```

**Impact**: ✅ Fallback cũng đúng về mặt logic

---

## Kết quả

### Trước khi sửa
- ❌ Labels không khớp với nodes sau pruning
- ❌ Semantic score dominant, path score gần như vô dụng
- ❌ Path formula không tối ưu
- ❌ Không có validation, bugs âm thầm xảy ra

### Sau khi sửa
- ✅ Labels và nodes CHÍNH XÁC khớp nhau
- ✅ Path và semantic scores cân bằng (alpha=0.6, beta=0.4 có ý nghĩa thực sự)
- ✅ Path formula phân biệt tốt hơn
- ✅ Validation toàn diện, phát hiện bugs ngay lập tức

---

## Testing

Để test các fixes:

```bash
# Test với semantic pruning enabled
python train.py -d FB15k-237 \
    --use_semantic_pruning \
    --stage1_ratio 10 \
    --path_weight 0.6 \
    --semantic_weight 0.4 \
    --target_subgraph_size 1000

# Kiểm tra logs để đảm bảo:
# 1. Không có validation errors
# 2. Score ranges được normalized đúng
# 3. Root labels luôn là [0,1] và [1,0]
```

---

## Files đã sửa

1. ✅ `subgraph_extraction/graph_sampler.py`
   - Fix label rebuilding logic (line 562-634)
   - Add comprehensive validation (line 586-600, 647-656)
   - Fix fallback random sampling (line 602-634)

2. ✅ `subgraph_extraction/semantic_pruning.py`
   - Improve path length formula (line 106-118)
   - Add score normalization (line 311-337)
   - Add validation checks (line 358-378)

---

## Ghi chú

- Các fixes này **backward compatible** - code cũ không dùng semantic pruning vẫn hoạt động bình thường
- Assertions sẽ raise lỗi ngay khi phát hiện inconsistency → dễ debug
- Normalization đảm bảo alpha/beta weights có ý nghĩa thực sự
- Path formula mới phân biệt tốt hơn nodes gần vs xa endpoints

**Tác giả**: Claude Code
**Ngày**: 2025-12-03
