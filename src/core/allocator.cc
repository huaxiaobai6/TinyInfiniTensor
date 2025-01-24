#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = this->getAlignedSize(size);
 
    // 查找合适的空闲块
    auto it = freeBlocks.begin();
    while (it != freeBlocks.end()) {
      if (it->second >= size) {
        size_t addr = it->first;
        if (it->second > size) {
          // 分割空闲块
          freeBlocks[addr + size] = it->second - size;
        }
        freeBlocks.erase(it);
        used += size;
        if (used > peak) {
          peak = used;
        }
        return addr;
      }
      ++it;
    }
 
    // 如果没有合适的空闲块，分配新内存
    if (this->ptr == nullptr) {
      this->ptr = runtime->alloc(size);
      printf("Allocator really alloc: %p %lu bytes\n", this->ptr, size);
    } else {
      // 需要扩展内存（简化处理，实际中可能需要更复杂的逻辑）
      // 这里假设我们总是重新分配以容纳新的峰值需求
      void *newPtr = runtime->alloc(peak + size);
      memcpy(newPtr, this->ptr, peak);
      runtime->dealloc(this->ptr);
      this->ptr = newPtr;
    }
    peak += size;
    used += size;
    return (size_t)((char*)this->ptr + (peak - size));  // 返回偏移量
  }

    void Allocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);
 
    // 插入或合并空闲块
    auto it = freeBlocks.lower_bound(addr);
    if (it != freeBlocks.begin() && std::prev(it)->first + std::prev(it)->second == addr) {
      // 与前一个块合并
      addr = std::prev(it)->first;
      size += std::prev(it)->second;
      freeBlocks.erase(std::prev(it));
    }
    if (it != freeBlocks.end() && addr + size == it->first) {
      // 与后一个块合并
      size += it->second;
      freeBlocks.erase(it);
    }
    freeBlocks[addr] = size;
    used -= size;
  }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
