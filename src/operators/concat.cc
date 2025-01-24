#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    if (inputs.empty()) {
        return std::nullopt; // 如果没有输入，返回无效值
    }
 
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();
 
    // 初始化拼接后的形状为第一个输入的形状
    Shape concatDims = dims;
 
    // 计算拼接维度上的总大小
    int concatDimSize = 0;
    for (const auto &input : inputs) {
        concatDimSize += input->getDims()[dim];
    }
 
    // 更新拼接维度的大小
    concatDims[dim] = concatDimSize;
 
    return {{concatDims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
