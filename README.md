# scbamtools

High performance Cython + Python tools to process BAM files with tags as they arise in single-cell sequencing

## Status

This is alpha. Mostly, the plan is to move useful functionality developed within [spacemake](https://github.com/rajewsky-lab/spacemake) outside of spacemake, so that it can be re-used without pulling in all the dependencies for a heavy-weight spatial transcriptomics package.
Currently, the umbilical has not been cut and the code is almost certainly not functional w/o spacemake around.

### Useful things include:

  * converting FASTQ files to uBAM files with barcode information (single cell and spatial workflows)
  * trimming adapters (uses cutadapt functions under the hood)
  * making histograms and statistics about cell barcodes, UMIs and possibly other BAM tags
  * annotate aligned BAM records against a transcript annotation such as GENCODE
  * build digital gene expression counts from annotated BAM files, directly as scanpy AnnData (h5ad)

## Why is this better than ...

Depends what you need. We are building these tools to be as fast as possible while keeping as much of the functionality in python (with the occasional cython) for felxibility and maintainability. We don't care as much about (total) CPU use as we care about throughput/scalability. So, some principles:

  * avoid temp files, streaming is better
  * parallelize with [mrfifo](https://github.com/marvin-jens/mrfifo) for low-overhead parallelism
  * put some effort into efficient data structures where it pays off
  * make simple things simple, while hard things should be possible

The code in here is the same that we use to process open-st spatial transcriptomics data, which is *very deep*: typical runs having billions of reads and hundreds of millions of spatial barcodes. While we make sure that the tools here don't break and have manageable resource usage, we do not intend to be the most CPU-efficient or allow you to process open-st on your laptop. YMMV.

## Roadmap

  * port everything from spacemake [ongoing]
  * full suite of tools to replace dropseq-tools in spacemake [v1.0]
  * optimizations
  * tutorials and example uses outside of spacemake




