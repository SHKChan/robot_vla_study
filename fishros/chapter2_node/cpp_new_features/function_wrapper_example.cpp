#include <iostream>
#include <algorithm>
#include <functional>

void save_with_func(const std::string &filename)
{
    std::cout << "Save with func: " << filename << std::endl;
}

void save_with_lambda(const std::string &filename)
{
    std::cout << "Save with lambda: " << filename << std::endl;
}

class FileSaver
{
public:
    void save_with_method(const std::string &filename)
    {
        std::cout << "Save with class: " << filename << std::endl;
    }
};

int main(int argc, char **argv)
{
    FileSaver fs;
    save_with_func("file1.txt");
    save_with_lambda("file2.txt");
    fs.save_with_method("file3.txt");

    // Use function wrapper
    std::function<void(const std::string &)> saveFunc1 = save_with_func;
    std::function<void(const std::string &)> saveFunc2 = save_with_lambda;
    std::function<void(const std::string &)> saveFunc3 = std::bind(&FileSaver::save_with_method, &fs, std::placeholders::_1);
    saveFunc1("file1.txt");
    saveFunc2("file2.txt");
    saveFunc3("file3.txt");
    return 0;
}