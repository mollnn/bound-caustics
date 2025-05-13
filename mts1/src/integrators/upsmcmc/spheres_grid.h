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

#if !defined(__SPHERES_GRID_H)
#define __SPHERES_GRID_H

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/math.h>
#include <functional>

MTS_NAMESPACE_BEGIN

template< typename T>
class SpheresGrid {
public:
    SpheresGrid() {
    }

    void clear() {
        m_cells.clear();
        m_indices.clear();
    }
    void build(const std::vector<T> & spheres) {
        m_spheres = &spheres;
        // Get bounding box
        m_min = Point(std::numeric_limits<Float>::max());
        m_max = -m_min;
        for_each(spheres.begin(), spheres.end(), [&](const T & sphere) {
            const Vector radius(sphere.radius);
            const Point min = sphere.position - radius;
            const Point max = sphere.position + radius;
            m_min.x = std::min(min.x, m_min.x);
            m_min.y = std::min(min.y, m_min.y);
            m_min.z = std::min(min.z, m_min.z);
            m_max.x = std::max(max.x, m_max.x);
            m_max.y = std::max(max.y, m_max.y);
            m_max.z = std::max(max.z, m_max.z);
        });
        // Enlarge by 1%
        const Vector one_percent = Vector(1.e-3f) + (m_max - m_min) * 0.01f;
        m_min -= one_percent;
        m_max += one_percent;
        // Prepare cells
        m_cellsPerDim = (int)pow((double)spheres.size(), 1.0 / 3.0) + 1;
        m_cellsPerDimSqr = m_cellsPerDim * m_cellsPerDim;
        m_cells.resize(m_cellsPerDim * m_cellsPerDimSqr + 1);
        memset(&m_cells[0], 0, m_cells.size() * sizeof(int));
        const Vector diff = m_max - m_min;
        m_cellSize = diff / (Float)m_cellsPerDim;
        m_inverseCell = Vector(1.f / m_cellSize.x, 1.f / m_cellSize.y, 1.f / m_cellSize.z);
        // Do first sweep to compute number of spheres in each cell
        rasterSpheres(spheres, [&](int cellIndex, int sphereIndex) { 
            /*if (!(0 <= cellIndex && cellIndex < m_cells.size()))
                SLog(EError, "FUCK %i >= %i", cellIndex, m_cells.size());*/
            ++m_cells[cellIndex]; 
        });
        // Now accumulate cell
        int m_hitCells = 0;
        for_each(m_cells.begin(), m_cells.end(), [&](int & cell) { 
            m_hitCells += cell; 
            cell = m_hitCells; 
        });
        if (m_hitCells > 1000000000 || m_hitCells < 0)
            SLog(EError, "Too many indices: %i",m_hitCells);
        // Finally fill the indices
        m_indices.resize(m_hitCells);
        memset(&m_indices[0], 0, m_indices.size() * sizeof(int));
        rasterSpheres(spheres, [&](int cellIndex, int sphereIndex) {
            /*if (!(0 <= cellIndex && cellIndex < m_cells.size()))
                SLog(EError, "FUCK %i >= %i", cellIndex, m_cells.size());*/
            int index = --m_cells[cellIndex];
            /*if (!(0 <= index && index < m_indices.size()))
                SLog(EError, "FUCK %i >= %i", index, m_indices.size());*/
            m_indices[index] = sphereIndex;
        });
        // Now indices are filled and m_indices[m_cells[i]] is the first sphere in the cell i.
    }

    bool find_any(const Point & p, const Normal & n) const {
        // Is p inside the grid?
        int cellIndex;
        if (!index_test(p, cellIndex))
            return false;
        // Goes through all spheres in a cell that contains p
        for (int i = m_cells[cellIndex], end = m_cells[cellIndex + 1]; i != end; ++i) {
            // Test intersection
            const T & sphere = (*m_spheres)[m_indices[i]];
            if (testSphere(p, sphere.position, sphere.radius * sphere.radius))
                return true;
        }
        return false;
    }

    void find_all(const Point & p, std::vector<int> & indices) const {
        // Is p inside the grid?
        int cellIndex;
        if (!index_test(p, cellIndex))
            return;
        // Goes through all spheres in a cell that contains p
        for (int i = m_cells[cellIndex], end = m_cells[cellIndex + 1]; i != end; ++i) {
            // Test intersection
            const T & sphere = (*m_spheres)[m_indices[i]];
            if (testSphere(p, sphere.position, sphere.radius * sphere.radius))
                indices.push_back(m_indices[i]);
        }
    }

    void query(const Point & p, const std::function<void(int index)>& functor) const {
        // Is p inside the grid?
        int cellIndex;
        if (!index_test(p, cellIndex))
            return;
        // Goes through all spheres in a cell that contains p
        for (int i = m_cells[cellIndex], end = m_cells[cellIndex + 1]; i != end; ++i) {
            // Test intersection
            const T & sphere = (*m_spheres)[m_indices[i]];
            if (testSphere(p, sphere.position, sphere.radius * sphere.radius))
                functor(m_indices[i]);
        }
    }

    /*void find_all_brute_force(const Point & p, std::vector<int> & indices) const {
        for (auto it = m_spheres->cbegin(); it != m_spheres->cend(); ++it) {
            if (testSphere(p, it->position, it->radius * it->radius))
                indices.push_back((int)(it - m_spheres->cbegin()));
        }
    }

    bool find_any(const Point & p) const {
        std::vector<int> i1, i2;
        find_all(p, i1);
        find_all_brute_force(p, i2);
        SAssert(i1.size() == i2.size());
        std::sort(i1.begin(), i1.end());
        std::sort(i2.begin(), i2.end());
        for (size_t i = 0; i < i1.size(); ++i) {
            SAssert(i1[i] == i2[i]);
        }
        return !i1.empty();
    }*/

private:

    template<typename Functor>
    void rasterSpheres(const std::vector<T> & spheres, Functor functor) {
        int sphereIndex = 0;
        for (auto it = spheres.cbegin(); it != spheres.cend(); ++it, ++sphereIndex) {
            const Vector radius(it->radius);
            const Float radiusSqr = it->radius * it->radius;
            const Point min = it->position - radius;
            const Point max = it->position + radius;
            Point3i imin, imax;
            getXYZClamp(min, imin);
            getXYZClamp(max, imax);
            int index = SpheresGrid::index(imin);
            int countx = imax.x - imin.x + 1;
            int county = (imax.y - imin.y + 1) * m_cellsPerDim;
            Vector cellStart(m_cellSize.x * (Float)imin.x + m_min.x, m_cellSize.y * (Float)imin.y + m_min.y, m_cellSize.z * (Float)imin.z + m_min.z);
            Vector cell = cellStart;
            for (int z = imin.z; z <= imax.z; ++z) {
                Float distZ = std::max(std::max(cell.z - it->position.z, it->position.z - (cell.z + m_cellSize.z)), (Float)0.f);
                distZ = distZ * distZ;
                cell.y = cellStart.y;
                for (int y = imin.y; y <= imax.y; ++y) {
                    cell.x = cellStart.x;
                    Float distY = std::max(std::max(cell.y - it->position.y, it->position.y - (cell.y + m_cellSize.y)), (Float)0.f);
                    distY = distY * distY + distZ;
                    for (int x = imin.x; x <= imax.x; ++x) {
                        Float distX = std::max(std::max(cell.x - it->position.x, it->position.x - (cell.x + m_cellSize.x)), (Float)0.f);
                        distX = distX * distX + distY;
                        //SAssert(0 <= index && index < m_cells.size());
                        if (distX <= radiusSqr)
                            functor(index, sphereIndex);
                        // Increase index in x
                        ++index; 
                        cell.x += m_cellSize.x;
                    }
                    // Increase index in y and decrease it in x
                    index += m_cellsPerDim - countx;
                    cell.y += m_cellSize.y;
                }
                index += m_cellsPerDimSqr - county; // Increase index in z and decrease it in y
                cell.z += m_cellSize.z;
            }
        }
    }

    inline bool testSphere(const Point &p, const Point & center, float radiusSqr) const {
        const Vector diff = p - center;
        return diff.lengthSquared() <= radiusSqr;
    }

    inline void getXYZ(const Point & p, Point3i & out) const {
        const Vector v = p - m_min;
        out.x = (int)(v.x * m_inverseCell.x);
        out.y = (int)(v.y * m_inverseCell.y);
        out.z = (int)(v.z * m_inverseCell.z);
    }

    inline void getXYZClamp(const Point & p, Point3i & out) const {
        getXYZ(p, out);
        out.x = math::clamp(out.x, 0, m_cellsPerDim - 1);
        out.y = math::clamp(out.y, 0, m_cellsPerDim - 1);
        out.z = math::clamp(out.z, 0, m_cellsPerDim - 1);
    }

    inline int index(const Point & p) const {
        Point3i xyz;
        getXYZ(p, xyz);
        return index(xyz);
    }

    // Returns true if p is inside the grid
    inline bool index_test(const Point & p, int & i) const {
        Point3i xyz;
        getXYZ(p, xyz);
        i = index(xyz);
        return (xyz.x >= 0 && xyz.x < m_cellsPerDim &&
            xyz.y >= 0 && xyz.y < m_cellsPerDim &&
            xyz.z >= 0 && xyz.z < m_cellsPerDim);
    }

    inline int index(const Point3i & xyz) const {
        return xyz.x + m_cellsPerDim * xyz.y + m_cellsPerDimSqr * xyz.z;
    }

    std::vector<int> m_indices;
    std::vector<int> m_cells;
    const std::vector<T> * m_spheres;
    Point m_min;
    Point m_max;
    int m_cellsPerDim;
    int m_cellsPerDimSqr;
    Vector m_inverseCell;
    Vector m_cellSize;
};

MTS_NAMESPACE_END

#endif /* __SPHERES_GRID */
