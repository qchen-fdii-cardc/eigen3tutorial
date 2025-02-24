#include <iostream>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;

int main()
{
     // 创建稀疏矩阵
     SparseMatrix<double> sparse(5, 5);
     vector<Triplet<double>> triplets;

     // 添加非零元素
     triplets.push_back(Triplet<double>(0, 0, 1.0));
     triplets.push_back(Triplet<double>(1, 1, 2.0));
     triplets.push_back(Triplet<double>(2, 2, 3.0));
     triplets.push_back(Triplet<double>(1, 2, 0.5));
     triplets.push_back(Triplet<double>(2, 1, 0.5));

     sparse.setFromTriplets(triplets.begin(), triplets.end());
     cout << "稀疏矩阵:\n"
          << MatrixXd(sparse) << "\n\n";

     // 稀疏矩阵运算
     SparseMatrix<double> squared = sparse * sparse;
     cout << "稀疏矩阵的平方:\n"
          << MatrixXd(squared) << "\n\n";

     // 求解稀疏线性系统
     VectorXd b(5);
     b << 1, 2, 3, 4, 5;

     SimplicialLDLT<SparseMatrix<double>> solver;
     solver.compute(sparse);
     if (solver.info() == Success)
     {
          VectorXd x = solver.solve(b);
          cout << "线性系统解:\n"
               << x << "\n\n";
     }

     return 0;
}