#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        const Shape &shapeA = inputs[0]->getShape();
    const Shape &shapeB = inputs[1]->getShape();
 
    // 确定 A 和 B 的有效形状
    size_t m, n, k;
    if (transA) {
        IT_ASSERT(shapeA.size() == 2, "Matrix A must be 2D for transposition.");
        m = shapeA[1]; // A^T 的行数
        n = shapeA[0]; // A^T 的列数
    } else {
        IT_ASSERT(shapeA.size() == 2, "Matrix A must be 2D.");
        m = shapeA[0];
        n = shapeA[1];
    }
 
    if (transB) {
        IT_ASSERT(shapeB.size() == 2, "Matrix B must be 2D for transposition.");
        k = shapeB[1]; // B^T 的列数
        IT_ASSERT(n == shapeB[0], "Incompatible shapes for matrix multiplication."); // A 的列数必须等于 B^T 的行数
    } else {
        IT_ASSERT(shapeB.size() == 2, "Matrix B must be 2D.");
        k = shapeB[1];
        IT_ASSERT(n == shapeB[0], "Incompatible shapes for matrix multiplication.");
    }
 
    // 结果形状是 [m, k]
    Shape resultShape = {m, k};
    return std::vector<Shape>{resultShape};
    }

} // namespace infini