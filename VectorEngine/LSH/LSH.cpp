#include <LSH.h>
#include <random>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

template<size_t hash_num, 
         size_t table_num, 
         size_t dim, 
         typename valT>
LSH::LSH() : vec_data(), table_data(), hash_functions()
{

    std::default_random_engine gen(2022);
    std::normal_distribution<valT> dist(0.0, 1.0);
    for(auto& hash_func : hash_functions)
        for(auto& val : hash_func)
            val = dist(gen);

    fmt::print("----- LSH Init Done -----\n");
}


template<size_t hash_num,
         size_t table_num,
         size_t dim,
         typename valT>
void LSH::read_query_vectors(const char* fs)
{
}

