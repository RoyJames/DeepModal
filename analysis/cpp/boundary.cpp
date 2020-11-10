#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdio>
namespace py = pybind11;

int dx[6] = {-1, 1, 0, 0, 0, 0};
int dy[6] = {0, 0, -1, 1, 0, 0};
int dz[6] = {0, 0, 0, 0, -1, 1};
void dfs(int *flood, int *flags, int res, int x_, int y_, int z_)
{
    // printf("%d %d %d\n", x_, y_, z_);
    auto flood_idx = [res](int x, int y, int z) { return (z + 1) * (res + 2) * (res + 2) + (y + 1) * (res + 2) + (x + 1); };
    flood[flood_idx(x_, y_, z_)] = 1;
    for (int i = 0; i < 6; i++)
    {
        int x = x_ + dx[i];
        int y = y_ + dy[i];
        int z = z_ + dz[i];
        if (x < -1 || x > res || y < -1 || y > res || z < -1 || z > res)
            continue;
        if (flood[flood_idx(x, y, z)])
        {
            continue;
        }
        if (x >= 0 && x < res && y >= 0 && y < res && z >= 0 && z < res && flags[z * res * res + y * res + x])
        {
            continue;
        }
        dfs(flood, flags, res, x, y, z);
    }
}
<<<<<<< HEAD
auto flood_dfs(py::array_t<int> flags, int res)
{
=======
auto flood_dfs(py::array_t<int> flags, int res){

    auto info_flags = flags.request();
    int flood_size = (res+2)*(res+2)*(res+2);
    auto result = py::array_t<int>(flood_size);
    int *flood = static_cast<int*>(result.request().ptr);
    memset(flood, 0, sizeof(int)*flood_size);
    dfs(flood, static_cast<int*>(info_flags.ptr), res, -1, -1, -1);
    return result;
}
>>>>>>> 9e91e9052ddc2f40996d02a5b6d3292290e83072

    auto info_flags = flags.request();
    int flood_size = (res + 2) * (res + 2) * (res + 2);
    auto result = py::array_t<int>(flood_size);
    int *flood = static_cast<int *>(result.request().ptr);
    memset(flood, 0, sizeof(int) * flood_size);
    dfs(flood, static_cast<int *>(info_flags.ptr), res, -1, -1, -1);
    return result;
}

auto create(py::array_t<int> flags, py::array_t<double> vertices, py::array_t<int> tets, int res)
{
    auto info_flags = flags.request();
    auto info_vertices = vertices.request();
    auto info_tets = tets.request();

    int flood_size = (res + 2) * (res + 2) * (res + 2);
    auto result = py::array_t<int>(flood_size);
    int *flood = static_cast<int *>(result.request().ptr);
    memset(flood, 0, sizeof(int) * flood_size);

    dfs(flood, static_cast<int *>(info_flags.ptr), res, -1, -1, -1);
    int *t = static_cast<int *>(info_tets.ptr);
    double *v = static_cast<double *>(info_vertices.ptr);

    // int cube_tets[5][4] = {
    //     {0,2,3,6},{0,3,1,5},{0,4, 6,5}, {5, 7, 6, 3},{3, 0, 5, 6},
    //    //0,1,2,3   4,5,6,7   8,9,10,11, 12,13,14,15,  16,17,18,19
    // };

    auto faces = std::vector<int>();
    auto check = [res, flood](int x, int y, int z) {
        return flood[(z + 1) * (res + 2) * (res + 2) + (y + 1) * (res + 2) + (x + 1)];
    };

    int i;
    auto apd = [&faces, t, &i](int v1, int v2, int v3) {
        faces.push_back(t[i * 20 + v1]);
        faces.push_back(t[i * 20 + v2]);
        faces.push_back(t[i * 20 + v3]);
    };
    for (i = 0; i < (info_tets.shape[0] / 5); i++)
    {
        int base = t[i * 20];
        int x = v[base * 3];
        int y = v[base * 3 + 1];
        int z = v[base * 3 + 2];
        if (check(x + 1, y, z))
        {
            apd(6, 5, 7);
            apd(12, 15, 13);
        }
        if (check(x - 1, y, z))
        {
            apd(0, 3, 1);
            apd(8, 9, 10);
        }
        if (check(x, y + 1, z))
        {
            apd(1, 3, 2);
            apd(13, 15, 14);
        }
        if (check(x, y - 1, z))
        {
            apd(4, 6, 7);
            apd(8, 11, 9);
        }
        if (check(x, y, z + 1))
        {
            apd(9, 11, 10);
            apd(12, 13, 14);
        }
        if (check(x, y, z - 1))
        {
            apd(0, 1, 2);
            apd(4, 5, 6);
        }
    }
    return faces;
}

PYBIND11_MODULE(boundary, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("create", &create, py::arg("flags"),
<<<<<<< HEAD
          py::arg("vertices"),
          py::arg("tets"),
          py::arg("res"));
    m.def("flood_dfs", &flood_dfs, py::arg("flags"), py::arg("res"));
=======
                        py::arg("vertices"),
                        py::arg("tets"),
                        py::arg("res"));
    m.def("flood_dfs", &flood_dfs, py::arg("flags"),py::arg("res"));
>>>>>>> 9e91e9052ddc2f40996d02a5b6d3292290e83072
}