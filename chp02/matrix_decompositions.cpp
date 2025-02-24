#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
    Matrix3d A;
    A << 4, -2, 2,
        -2, 10, -6,
        2, -6, 8;

    cout << "原始矩阵 A:\n"
         << A << "\n\n";

    // LU分解
    auto lu = A.fullPivLu();
    Matrix3d L = Matrix3d::Identity();
    L.triangularView<StrictlyLower>() = lu.matrixLU();
    Matrix3d U = lu.matrixLU().triangularView<Upper>();
    cout << "LU分解 L:\n"
         << L << "\n\n";
    cout << "LU分解 U:\n"
         << U << "\n\n";

    // QR分解
    auto qr = A.householderQr();
    Matrix3d Q = qr.householderQ() * Matrix3d::Identity();
    Matrix3d R = qr.matrixQR().triangularView<Upper>();
    cout << "QR分解 Q:\n"
         << Q << "\n\n";
    cout << "QR分解 R:\n"
         << R << "\n\n";

    // Cholesky分解
    auto llt = A.llt();
    Matrix3d L_chol = llt.matrixL();
    cout << "Cholesky分解 L:\n"
         << L_chol << "\n\n";

    // SVD分解
    auto svd = A.jacobiSvd(ComputeFullU | ComputeFullV);
    cout << "奇异值:\n"
         << svd.singularValues() << "\n\n";
    cout << "左奇异向量 U:\n"
         << svd.matrixU() << "\n\n";
    cout << "右奇异向量 V:\n"
         << svd.matrixV() << "\n\n";

    return 0;
}