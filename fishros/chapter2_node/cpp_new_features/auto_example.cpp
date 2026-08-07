#include <iostream>

int main(int argc, char **argv)
{
    auto intA = 10;
    auto doubleB = 3.14;
    auto chartC = 'f';

    std::cout << "A: " << typeid(intA).name() << std::endl;
    std::cout << "B: " << typeid(doubleB).name() << std::endl;
    std::cout << "C: " << typeid(chartC).name() << std::endl;
}