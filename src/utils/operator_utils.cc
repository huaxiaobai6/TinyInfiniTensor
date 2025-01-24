#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    size_t rankA = A.size();
    size_t rankB = B.size();
 
    // 确定哪个形状的秩更大
    Shape broadcastShape;
 
    // 如果 A 的秩较小，在前面添加 1
    if (rankA < rankB) {
        broadcastShape.insert(broadcastShape.end(), rankB - rankA, 1);
        broadcastShape.insert(broadcastShape.end(), A.begin(), A.end());
    } 
    // 如果 B 的秩较小，在前面添加 1
    else if (rankA > rankB) {
        broadcastShape.insert(broadcastShape.end(), rankA - rankB, 1);
        broadcastShape.insert(broadcastShape.end(), B.begin(), B.end());
    } 
    // 如果秩相等，直接使用 A（或 B）的大小
    else {
        broadcastShape = A; // 或 B，因为它们的大小相同
    }
 
    // 检查兼容性并计算广播后的形状
    Shape resultShape;
    for (size_t i = 0; i < std::max(rankA, rankB); ++i) {
        size_t dimA = (i < rankA) ? A[i] : 1;
        size_t dimB = (i < rankB) ? B[i] : 1;
 
        // 确保维度兼容
        IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1,
                  "Shapes are not broadcastable: " << A << " and " << B);
 
        // 取最大值作为广播后的维度大小
        resultShape.push_back(std::max(dimA, dimB));
    }
 
    return resultShape;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
