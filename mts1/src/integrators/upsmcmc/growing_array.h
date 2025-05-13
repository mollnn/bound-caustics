/*
This file is part of Mitsuba, a physically based rendering system.

Copyright (c) 2007-2014 by Wenzel Jakob and others.

Mitsuba is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 3
as published by the Free Software Foundation.

Mitsuba is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#if !defined(__MITSUBA_GROWING_ARRAY_H_)
#define __MITSUBA_GROWING_ARRAY_H_

#include <mitsuba/mitsuba.h>
#include <vector>

MTS_NAMESPACE_BEGIN

#if 1
// Array that can only grow (or be deallocated as a whole)
template<typename T>
class GrowingArray {
public:
    
    GrowingArray(size_t allocSize = 2048):
        m_allocSize(allocSize),
        m_free(0) {
    }

    size_t size() const {
        return m_allocated.size() * m_allocSize - m_free;
    }

    void push_back(const T & t) {
        if (EXPECT_NOT_TAKEN(!m_free)) {
            m_free = m_allocSize;
            m_allocated.push_back(new T[m_allocSize]);
        }
        m_allocated.back()[m_allocSize - m_free] = t;
        --m_free;
    }

    void insert_coherent(const T * t, size_t size) {
        if (EXPECT_NOT_TAKEN(m_free < size)) {
            m_free = m_allocSize;
            m_allocated.push_back(new T[m_allocSize]);
        }
        memcpy(m_allocated.back() + (m_allocSize - m_free), t, sizeof(T)* size);
        m_free -= size;
    }

    // Performs move of the memory by adding memory blocks from <other>. The currently unfinished memory block will
    // remain in the middle with uninitialized T items.
    // Returns the starting index of the added items;
    size_t move_effective(GrowingArray & other) {
        SAssert(other.m_allocSize == m_allocSize);
        size_t current = m_allocated.size() * m_allocSize;
        m_allocated.insert(m_allocated.end(), other.m_allocated.begin(), other.m_allocated.end());
        m_free = other.m_free;
        other.m_allocated.clear();
        other.m_free = 0;
        return current;
    }

    T & operator[](size_t index) {
        SAssert(index / m_allocSize < m_allocated.size());
        return m_allocated[index / m_allocSize][index % m_allocSize];
    }

    const T & operator[](size_t index) const {
        SAssert(index / m_allocSize < m_allocated.size());
        return m_allocated[index / m_allocSize][index % m_allocSize];
    }

    void dealloc() {
        for (T * t : m_allocated) {
            delete[] t;
        }
        m_allocated.clear();
        m_free = 0;
    }

    ~GrowingArray() {
        dealloc();
    }
private:
    // Allocated blocks
    std::vector<T *> m_allocated;
    // Block size
    size_t m_allocSize;
    // Free space in the last block
    size_t m_free;
};

#else
// Array that can only grow (or be deallocated as a whole)
template<typename T>
class GrowingArray {
public:

    GrowingArray(size_t allocSize = 2048){}

    size_t size() const {
        return m_allocated.size();
    }

    void push_back(const T & t) {
        m_allocated.push_back(t);
    }

    // Performs move of the memory by adding memory blocks from <other>. The currently unfinished memory block will
    // remain in the middle with uninitialized T items.
    // Returns the starting index of the added items;
    size_t move_effective(GrowingArray & other) {
        size_t current = m_allocated.size();
        m_allocated.insert(m_allocated.end(), other.m_allocated.begin(), other.m_allocated.end());
        other.m_allocated.clear();
        return current;
    }

    T & operator[](size_t index) {
        if (index >= m_allocated.size())
            SLog(EError, "Out of bounds");
        return m_allocated[index];
    }

    const T & operator[](size_t index) const {
        if (index >= m_allocated.size())
            SLog(EError, "Out of bounds");
        return m_allocated[index];
    }

    void dealloc() {
        m_allocated.clear();
    }

    ~GrowingArray() {
    }
private:
    // Allocated blocks
    std::vector<T> m_allocated;
};
#endif
MTS_NAMESPACE_END

#endif /* __MITSUBA_GROWING_ARRAY_H_ */