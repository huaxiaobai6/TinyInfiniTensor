#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        int rank = A->getRank();
 
        // 初始化输出维度为输入维度的大小
        std::vector<int> output_dim(rank);
 
        // 根据 permute 重新排列维度
        for (int i = 0; i < rank; ++i)
        {
            output_dim[i] = input_dim[transposePermute[i]];
        }
 
        // 创建包含输出形状的向量
        std::vector<Shape> output_shapes;
        output_shapes.emplace_back(output_dim);
 
        // 返回输出形状
        return output_shapes;
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
