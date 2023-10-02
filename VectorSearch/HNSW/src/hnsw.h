#include <vector>
#include <tuple>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <random>
#include <cstring>

namespace MTHierarchicalNSW{

typedef uint64_t idT;
typedef uint64_t sizeT;

template<typename dataT, typename labelT, size_t dim>
class MTHierarchicalNSW
{
public:
    MTHierarchicalNSW(size_t efM, size_t maxM, size_t maxN) : efM_(efM), maxM_(maxM), maxN_(maxN)
    {
        this->size_per_data_ = sizeof(sizeT) + sizeof(idT) * maxM_ * 2 + sizeof(dataT) * dim + sizeof(labelT);
        this->size_per_link_ = sizeof(sizeT) + sizeof(idT) * maxM_;
        this->data_offset_ = sizeof(sizeT) + sizeof(idT) * maxM_ * 2;
        this->label_offset_ = sizeof(sizeT) + sizeof(idT) * maxM_ * 2 + sizeof(dataT) * dim;

        this->data_level_zero = (char*)malloc(size_per_data_ * maxN_);
        this->data_level_other = (char**)malloc(sizeof(void*) * maxN_);
    }

    const dataT fstdist(const dataT* d1, const dataT* d2) const ;
    void add_point(const dataT* data, idT label);

    void searchLayerSimple(idT& enterId, const idT& curId, const dataT* data, const size_t& level) const;
    void searchLayerEF(auto& ef_candidates, idT& enterId, const idT& curId, const dataT* data, const size_t& level) const;
    void connectNeighbors(auto& candidates, idT& enterId) const;

private:
    idT* getLabel(const idT& id) const;
    idT* getLabel(const idT* id) const;
    const sizeT getSize(const idT& id, size_t level) const;
    const sizeT getSize(const idT* id, size_t level) const;
    dataT* getData(const idT& id) const;
    dataT* getData(const idT* id) const;
    const idT* getNeighbor(const idT& id, size_t level) const;

private:
    size_t efM_, maxM_, maxN_;
    size_t size_per_data_, size_per_link_; 
    size_t data_offset_, label_offset_; 

    char *data_level_zero;
    char **data_level_other;
    std::unordered_map<labelT, idT> labelMap;
    std::unordered_map<idT, size_t> levelMap;

    idT enterId_ = 0;
    size_t maxL_ = 0;
};

template<typename dataT, typename labelT, size_t dim>
const dataT MTHierarchicalNSW<dataT, labelT, dim>::fstdist(const dataT* d1, const dataT* d2) const
{
    dataT ans;
    for(int i=0; i<dim; ++i)
        ans += d1[i] * d2[i];

    return ans;
}

template<typename dataT, typename labelT, size_t dim>
const idT* MTHierarchicalNSW<dataT, labelT, dim>::getNeighbor(const idT& id, size_t level) const
{
    if (level == 0)
        return reinterpret_cast<idT*>(data_level_zero + id * size_per_data_ + sizeof(size_t));
    else
        return reinterpret_cast<idT*>(data_level_other[id] + (level-1) * size_per_link_);
}


template<typename dataT, typename labelT, size_t dim>
const sizeT MTHierarchicalNSW<dataT, labelT, dim>::getSize(const idT& id, size_t level) const
{
    if (level == 0)
        return *reinterpret_cast<size_t*>(data_level_zero + id * size_per_data_);
    else
        return *reinterpret_cast<size_t*>(data_level_other[id] + (level-1) * size_per_link_);
}

template<typename dataT, typename labelT, size_t dim>
dataT* MTHierarchicalNSW<dataT, labelT, dim>::getData(const idT& id) const
{
    return reinterpret_cast<dataT*>(data_level_zero + id * size_per_data_ + data_offset_);
}

template<typename dataT, typename labelT, size_t dim>
dataT* MTHierarchicalNSW<dataT, labelT, dim>::getData(const idT* id) const
{
    return getData(*id);
}

template<typename dataT, typename labelT, size_t dim>
const sizeT MTHierarchicalNSW<dataT, labelT, dim>::getSize(const idT* id, size_t level) const
{
    return getSize(*id, level);
}

template<typename dataT, typename labelT, size_t dim>
idT* MTHierarchicalNSW<dataT, labelT, dim>::getLabel(const idT& id) const
{
    return reinterpret_cast<idT*>(data_level_zero + id * size_per_data_ + label_offset_);
}

template<typename dataT, typename labelT, size_t dim>
idT* MTHierarchicalNSW<dataT, labelT, dim>::getLabel(const idT* id) const
{
    return getLabel(*id);
}



template<typename dataT, typename labelT, size_t dim>
void MTHierarchicalNSW<dataT, labelT, dim>::add_point(const dataT* data, idT label)
{
    // label initialize
    size_t curL = -log(rand() /double(RAND_MAX)) / log(maxM_);
    idT curId = labelMap.size();
    levelMap[curId] = curL;
    labelMap.emplace(label, curId);
    printf("maxL_:%zu\n", this->maxL_);

    // level zero - initialize 
    memset(data_level_zero + curId * size_per_data_, 0, size_per_data_);
    memcpy(getLabel(curId), &label, sizeof(idT));
    memcpy(getData(curId), data, dim * sizeof(dataT));
    if(curL){
        // level other - initialize
        data_level_other[curId] = (char*)malloc(size_per_link_ * curL);
        memset(data_level_other[curId], 0, size_per_link_ * curL);
    } 

    idT enterId = this->enterId_;
    if(enterId_ != 0){
        // graph insert 
        for(auto i = maxL_ ; i > curL; --i)
            this->searchLayerSimple(enterId, curId, data, i);

        std::priority_queue<std::pair<dataT, idT>> ef_candidates;
        for(auto i = curL; i >= 0; --i){
            this->searchLayerEF(ef_candidates, enterId, curId, data, i);
            this->connectNeighbors(ef_candidates, enterId);
        }
    } else {
        enterId_ = 0;
        maxL_ = curL;
    }

    if(curL > this->maxL_ ) {
        this->enterId_ = curId;
        this->maxL_ = curL; 
    }

}



/*
template<typename dataT, typename labelT, size_t dim>
void MTHierarchicalNSW<dataT, idT>::searchLayerHEUR(const std::priority<std::pair<dataT, idT>>& ef_candidates, const size_t&& maxM, const size_t& level) const 
{
    if(ef_candidates.size() < maxM) 
        return ef_candidates;

    std::priority_queue<std::pair<dataT, idT>> he_candidates;
    while(!ef_candidates.empty()){
        he_candidates.emplace(-ef_candidates.top().first, ef_candidates.top().second);
        ef_candidates.pop();
    }

    while(!he_candidates.empty()){
    }
}
*/

template<typename dataT, typename labelT, size_t dim>
void MTHierarchicalNSW<dataT, labelT, dim>::searchLayerEF(
    auto& ef_candidates, 
    const idT& enterId, 
    const idT& curId, 
    const dataT* data, 
    const size_t& level
) const {
    std::priority_queue<std::pair<dataT, idT>> candidates;
    std::unordered_set<idT> visitedSet(maxN_);

    dataT D1 = fstdist(getData(enterId), data);
    visitedSet.insert(enterId);
    ef_candidates.emplace(D1, enterId);
    candidates.emplace(-D1, enterId);

    while(!candidates.empty()){
        std::pair<dataT, idT> top = candidates.top();
        if(-top.first > D1 && ef_candidates.size() == efM_) break;
        candidates.pop();

        const auto& datal = this->getNeighbor(top.second, level);
        const auto& size = this->getSize(enterId, level);
        for(int i=0; i<size; ++i){
            const auto& candidateId = *(datal + i);
            if(auto iter = visitedSet.find(candidateId); iter == visitedSet.end()){
                visitedSet.insert(candidateId);
                if(const auto& D2 = fstdist(getData(candidateId), data); ef_candidates.size() < efM_ || D2 < D1){
                    candidates.emplace(-D2, candidateId);
                    ef_candidates.emplace(D2, candidateId);
                    if(ef_candidates.size() > efM_) 
                        ef_candidates.pop();
                    D1 = ef_candidates.top().first;
                }
            }
        }
    } // while

}

template<typename dataT, typename labelT, size_t dim>
void MTHierarchicalNSW<dataT, labelT, dim>::searchLayerSimple(
    idT& enterId, 
    const idT& curId, 
    const dataT* data, 
    const size_t& level
) const {
    dataT D1 = fstdist(getData(enterId), data);
    bool changed = false;

    while(1){
        const auto& datal = this->getNeighbor(enterId, level);
        const auto& size = this->getSize(enterId, level);
        for(int i=0; i<size; ++i){
            const auto& candidateId = *(datal+i);
            if(const auto& D2 = fstdist(getData(candidateId), data); D2  < D1){
                enterId = candidateId;
                D1 = D2;
                changed = true;
            }
        }

        if(!changed) break;
    }
}

};
