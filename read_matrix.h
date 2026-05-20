// read_matrix.h
#pragma once

#include "common.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

inline bool read_matrix_market_to_csr(
    const std::string& filename,
    int& nrows, int& ncols,
    std::vector<int>& rowPtr,
    std::vector<int>& colInd,
    std::vector<VALUE_TYPE>& vals)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Cannot open " << filename << std::endl;
        return false;
    }

    std::string line;
    // banner
    if (!std::getline(fin, line)) {
        std::cerr << "Empty file\n";
        return false;
    }
    if (line.find("MatrixMarket") == std::string::npos) {
        std::cerr << "Not a MatrixMarket file\n";
        return false;
    }

    // skip comments
    do {
        if (!std::getline(fin, line)) {
            std::cerr << "Missing size line\n";
            return false;
        }
    } while (!line.empty() && line[0] == '%');

    int m, n, nnz;
    {
        std::istringstream iss(line);
        iss >> m >> n >> nnz;
        if (!iss) {
            std::cerr << "Invalid size line in mtx\n";
            return false;
        }
    }

    nrows = m;
    ncols = n;

    struct Trip { int r, c; VALUE_TYPE v; };
    std::vector<Trip> trips;
    trips.reserve(nnz);

    for (int k = 0; k < nnz; ++k) {
        int r, c;
        VALUE_TYPE v;
        fin >> r >> c >> v;
        if (!fin) {
            std::cerr << "Error reading entry " << k << "\n";
            return false;
        }
        r--; c--; // 1-based -> 0-based
        trips.push_back({r, c, v});
    }

    std::sort(trips.begin(), trips.end(),
              [](const Trip& a, const Trip& b){
                  if (a.r != b.r) return a.r < b.r;
                  return a.c < b.c;
              });

    rowPtr.assign(nrows + 1, 0);
    colInd.resize(nnz);
    vals.resize(nnz);

    int curRow = 0;
    int idx = 0;
    for (auto& t : trips) {
        while (curRow < t.r) {
            rowPtr[curRow + 1] = idx;
            ++curRow;
        }
        colInd[idx] = t.c;
        vals[idx]   = t.v;
        ++idx;
    }
    while (curRow < nrows) {
        rowPtr[curRow + 1] = idx;
        ++curRow;
    }
    return true;
}

void ilu_b(int m, VALUE_TYPE **b_add){
    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);
    for(int i=0; i<m; i++){
       b[i] = 1.0;
    }
    *b_add = b;
}