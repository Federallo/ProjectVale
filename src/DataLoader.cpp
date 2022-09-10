#include "DataLoader.h"

void DataLoader::readCSV(std::string filename, std::vector<std::unique_ptr<Eigen::RowVectorXf>>& data)
{
    data.clear();
    std::ifstream file(filename);
    std::string line, word;
    //determine number of columns in file
    std::getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<float> parsed_vec;
    while(std::getline(ss, word, ','))//FIXME ", "
    {
        parsed_vec.push_back(float(std::stof(&word[0])));
    }
    uint cols = parsed_vec.size();
    data.push_back(std::make_unique<Eigen::RowVectorXf>(cols));
    for(uint i = 0; i < cols; i++)
    {
        data.back() -> coeffRef(1, i) = parsed_vec[i];
    }

    //reading file
    if(file.is_open())
    {
        while(std::getline(file, line, '\n'))
        {
            std::stringstream ss(line);
            data.push_back(std::make_unique<Eigen::RowVectorXf>(1, cols));
            uint i = 0;
            while(std::getline(ss, word, ','))//FIXME ', '
            {
                data.back() -> coeffRef(i) = float(std::stof(&word[0]));
                i++;
            }
        }
    }
}

void DataLoader::generateData(std::string filename)
{
    std::ofstream file1(filename + "-in");
    std::ofstream file2(filename + "-out");
    for(uint i = 0; i < 1000; i++)
    {
        float x = std::rand()/float(RAND_MAX);
        float y = std::rand()/float(RAND_MAX);
        file1 << x << ", " << y << std::endl;
        file2 << 2*x + 10 + y << std::endl;
    }
    file1.close();
    file2.close();
}