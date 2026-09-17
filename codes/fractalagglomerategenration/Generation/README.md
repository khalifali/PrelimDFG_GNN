# Agglomerate Generation Scripts This folder contains two of the three
generation algorithms investigated in this work, namely the particle
cluster (PC) method and the local filling factor method.

Both scripts are used in essentially the same way. They take an XML
configuration file containing all required generation parameters such
as the particle radius, target particle count and algorithm specific
parameters. This approach was deliberately chosen to simplify batch
generation of agglomerates without having to modify the source code
whenever a parameter is changed.

```bash python PC.py <config.xml> ```
```bash python localFillingFactor.py <config.xml> ```


## Output File Format The scripts create a CSV file, that has no
header and contains the following columns : `x, y, z, r`,
corresponding to the center coordinates of the particle and the
radius.  These can be used to visualize or analyze the agglomerates
using third party software.

A very primitive way for plotting is also implemented in each script
but deactivated by default.


## LIGGGHTS Input Files For using the generated agglomerate files in
LIGGGHTS for DEM simulation, they have to be converted to a
proprietary LIGGGHTS Input File Format (`.sys`).

Converting the CSV files can be done using the liggghtsWriter script:
```bash python liggghtsWriter <input.csv> ```


# Aggglomerate Batch Generation Using the provided scripts,
`run_batch_lf.sh` and `run_batch_pc.sh`, multiple agglomerates can be
produced easily with different parameters. The parameters have to be
defined in the top part of each script and are used to generate a set
of config files for the algorithms, which are then subsequently run
and if chosen, converted to LIGGHTS input .sys files.

## Generating LIGGHTS Input Files Passing the `--sys` argument
converts all generated CSV files into LIGGGHTS .sys files.

```bash ./run_batch_pc.sh --sys ```


The converter automatically processes every generated CSV file and
writes the resulting .sys files to the corresponding output directory,
this can only be used after generating the CSV files.

## Cleaning generated files To remove all generated configuration
files, CSV files and .sys files, simply run

```bash ./run_batch_pc.sh --clean ```

The local filling factor batch script supports the same command line
arguments.


