#include <iostream>
#include <memory>

int main(int argc, char **argv)
{
    auto p1 = std::make_shared<std::string>("This is a str.");
    std::cout << "P1 ref count: " << p1.use_count() << ", at: " << p1 << std::endl
              << std::endl;

    auto p2 = p1;
    std::cout << "P1 ref count: " << p1.use_count() << ", at: " << p1 << std::endl;
    std::cout << "P2 ref count: " << p1.use_count() << ", at: " << p2 << std::endl
              << std::endl;

    p1.reset();
    std::cout << "P2 ref count: " << p1.use_count() << ", at: " << p2 << std::endl;
    std::cout << "P2 value: " << *p2 << std::endl;
}