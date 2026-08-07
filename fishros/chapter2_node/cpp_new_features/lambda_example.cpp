#include <iostream>
#include <algorithm>

int main(int argc, char **argv)
{
    auto intAdd = [](const int &a, const int &b) -> int
    {
        return a + b;
    };
    auto sum = intAdd(100, 50);

    auto printIntAdd = [sum]() -> void
    {
        std::cout << "Sum: " << sum << std::endl;
    };
    printIntAdd();
}