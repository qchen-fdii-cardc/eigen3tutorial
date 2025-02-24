#include <iostream>
#include <Eigen/Core>

int main()
{
    std::cout << "Eigen version: "
              << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    return 0;
}

/*
重要的宏定义：
1. EIGEN_WORLD_VERSION：主版本号
   - 表示不兼容的重大更改
   - Eigen3 与 Eigen2 就是不兼容的

2. EIGEN_MAJOR_VERSION：次版本号
   - 表示功能性更新
   - 通常保持向后兼容
   - 例如 3.4 添加了新的优化和特性

3. EIGEN_MINOR_VERSION：修订号
   - 表示 bug 修复和小改进
   - 不会改变 API
   - 如 3.4.1 到 3.4.2
*/