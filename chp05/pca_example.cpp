#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     // 创建示例数据
     MatrixXd data(100, 3);
     for (int i = 0; i < 100; ++i)
     {
          data.row(i) = Vector3d::Random();
     }

     cout << "原始数据前5行:\n"
          << data.topRows(5) << "\n\n";

     // 数据中心化
     MatrixXd centered = data.rowwise() - data.colwise().mean();
     cout << "中心化后数据前5行:\n"
          << centered.topRows(5) << "\n\n";

     // 计算协方差矩阵
     MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);
     cout << "协方差矩阵:\n"
          << cov << "\n\n";

     // 计算特征值和特征向量
     SelfAdjointEigenSolver<MatrixXd> solver(cov);
     cout << "特征值:\n"
          << solver.eigenvalues() << "\n\n";
     cout << "特征向量:\n"
          << solver.eigenvectors() << "\n\n";

     // 投影到主成分空间
     MatrixXd pc = centered * solver.eigenvectors();
     cout << "投影后数据前5行:\n"
          << pc.topRows(5) << "\n\n";

     return 0;
}