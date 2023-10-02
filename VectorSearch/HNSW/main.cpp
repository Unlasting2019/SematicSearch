#include <random>
#include <map>
#include <fstream>
#include <iostream>
#include <string>

#include "src/hnsw.h"

#define maxN 100
#define maxM 16
#define efM 64
#define dim 768

int main(){
    MTHierarchicalNSW::MTHierarchicalNSW<float, size_t, dim> s(efM, maxM, maxN);

    std::map<int, std::string> idMap;
    float* data = new float[maxN * dim];

    std::ifstream ifs("/home/test-t/SematicSearch/data/news_vector");
    std::string buf;

    size_t labelId = 0;
    while(std::getline(ifs, buf, '\n')){
        const char* start = buf.c_str(), *end = start + buf.size(), *sep = start;
        while(*sep != '\t') sep++;
        idMap[labelId++] = std::string(start, sep);

        size_t i = 0;
        for(const char *p1=sep+1, *p2=p1; p2 != end; ){
            while(*p2 != ';' && p2 != end) p2++;
            data[labelId*dim + i++] = std::stof(std::string(p1, p2));
            if(p2 == end) break;
            p1 = ++p2;
        }
        if(labelId == maxN) break;
    }

}
