# Welcome to the SharPy wiki!

### Update 2025/01/31

Files Changed:
`SAXScraft.py`: Major refactor and added new functions for generating voxel structures, assembling lattices, and improved visualization. Added new examples and flexible lattice/grid handling. Replace the old 'SASCraft.py'.

Code refactoring and organization:
- Reorganized the entire code by grouping related functions and added helper functions for easier readability and maintenance.
- Streamlined the code structure.
New functionality:
- `generate_voxel_structure()`: Added a new function to create various 3D voxel structures (e.g., Sphere, Trimer, Donut, etc.). It now supports additional geometries, including Helix structures.
- `generate_lattice_structure_3D()`: Added a new function to generate 3D lattices by repeating a structure (e.g., Sphere, Trimer) across a grid. The object shape, size, and other parameters can now be passed dynamically via arrays.
- Provided several examples for creating 2D or 3D assembled structures, including typical fcc, bcc, A15, and gammabrass lattice grids for easier use.
- Added the ability to handle periodic lattice structures.
Visualization improvements:
- Enhanced plotting capabilities with interactive sliders to visualize density and autocorrelation slices in 2D, replacing the previous slow Axes3D plotting approach for better performance.

### Update 2023/04/07

* The document was generated by ChatGPT. Please note that the content may not be completely accurate or relevant to your needs.
* SharPy is a Python script that performs a deconvolution of the Particle Pair Distribution Function (PDDF) in small-angle scattering experiments. It allows users to easily specify input files and tune the parameters.

## SharPy Documents

`SharPy` is a Python script that performs a deconvolution of the Particle Pair Distribution Function (PDDF) in small-angle scattering experiments. In brief, `SharPy` uses a single-particle PDDF as an initial guess and then optimizes the PDDF by minimizing the difference between the measured PDDF and the synthetic PDDF.

## Usage

To use `SharPy`, a user needs to specify an input file and some parameters.

### Input File

The input file should be in a binary format and contain a PDDF that was measured by a small-angle scattering experiment. Currently, `SharPy` supports two file formats: `out` and `pickle`. In the `out` format, the file should have two columns separated by a space or a tab, where the first column is the distance and the second column is the PDDF. In the `pickle` format, the file should be a binary file that contains a single numpy array containing the PDDF.

### Parameters

The following parameters can be tuned in `SharPy`:

- `method`: The optimization method used to minimize the difference between the measured PDDF and the synthetic PDDF. The default method is `BFGS`.
- `mode`: The speed of the optimization process. There are three options: `fast`, `medium`, and `slow`. The default mode is `slow`.
- `fend`: The file format of the input file. The two options are `out` and `pickle`. The default is `out`.
- `s`: The standard deviation of the size distribution. The default is `0.25`.
- `m`: The mean of the size distribution. The default is `50`.
- `dist`: The type of the size distribution. The two options are `lognorm` and `normal`. The default is `lognorm`.
- `R_size`: The number of data points used for the single-particle PDDF. The default is `51`.
- `save`: Whether to save the optimized PDDF and optimization results. The default is `True`.
- `output_fname`: The name of the output file. If not specified, a default name is generated.

## Example

Here is an example of how to use `SharPy`:

```
# read the input file
input_fname = 'example.out'

# set the size distribution parameters
s = 0.15
m = 50
dist = 'lognorm'

# set the number of data points used for the single-particle PDDF
R_size = 51

# run the optimization
res = SharPy.optimize(pdf_sync, s, m, dist, R_size)

# save the optimized PDDF and optimization results
output_fname = 'example_optimized'
SharPy.save(res, output_fname)

```

In this example, `SharPy` reads the PDDF from the `example.out` file, sets the size distribution parameters and the number of data points used for the single-particle PDDF, and then runs the optimization. The optimization results are saved in the `example_optimized` file.
