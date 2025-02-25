#include <Eigen/Dense>
#include <iostream>

int main()
{
    // creation
    Eigen::Matrix<double, 3, 3> A;
    Eigen::Matrix<double, 3, Eigen::Dynamic> B;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> E;

    Eigen::Matrix3f P, Q, R;
    Eigen::Vector3f x, y, z;
    Eigen::RowVector3f a, b, c;
    Eigen::VectorXd v;
    double s;

    A << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    std::cout << "A = \n" << A << '\n';


    // change B's cols to 3
    B.resize(3, 9);
    B << A, A, A;
    std::cout << "B = \n" << B << '\n';
    std::cout << "B.size() = " << B.size() << '\n';
    std::cout << "B.rows() = " << B.rows() << '\n';
    std::cout << "B.cols() = " << B.cols() << '\n';

    A.fill(1.0);
    std::cout << "A = \n" << A << '\n';

    // Basic Usage
    std::cout << "x.size() = " << x.size() << '\n'; // length(x)
    int i = 0;
    std::cout << "x(i) = " << x(i) << '\n'; // x(i+1)

    C = Eigen::MatrixXd::Random(5, 5);
    std::cout << "C.rows() = " << C.rows() << '\n'; // size(C, 1)
    std::cout << "C.cols() = " << C.cols() << '\n'; // size(C, 2)
    int j = 1;
    std::cout << "C = \n" << C << '\n';

    std::cout << "I = \n" << Eigen::Matrix3d::Identity() << '\n';

    C.setIdentity();
    std::cout << "C = \n" << C << '\n';

    C.setZero();
    std::cout << "C = \n" << C << '\n';

    C.setRandom();
    std::cout << "C = \n" << C << '\n';

    C.setConstant(1.0);
    std::cout << "C = \n" << C << '\n';

    C.setOnes();
    std::cout << "C = \n" << C << '\n';

    v.setLinSpaced(11, .1, 1.0);
    std::cout << "v = \n" << v << '\n';

    x.setLinSpaced(0, 1);
    std::cout << "x = \n" << x << '\n';
}
