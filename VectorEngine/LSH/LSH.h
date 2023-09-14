#pragma once
#include <array>
#include <vector>
#include <math.h>


template<size_t hash_num, 
         size_t table_num, 
         size_t dim,
         typename valT>
class LSH
{
private:
    using vecT = std::array<valT, dim>;
    std::vector<vecT> vec_data;
    std::array<std::vector<vecT>*, (size_t)std::pow(2, hash_num) * table_num> table_data;
    std::array<std::array<valT, dim>, hash_num * table_num> hash_functions;
public:
    LSH(); 
    void read_doc_vectors(const char* fs);
    void read_query_vectors(const char* fs);
};
