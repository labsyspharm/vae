import os
import argparse
import pandas as pd
import scanpy as sc
import anndata as ad


def parseArgs():
    '''
    Parse arguments.
    Input file is required.
    '''
    parser = argparse.ArgumentParser(
        description='Cluster cell types using latent '
                    'vectors of pixel image patches.'
    )
    parser.add_argument(
        '-i', '--input', help='Input CSV of latent vector data for cells',
        type=str, required=True
    )
    parser.add_argument(
        '-o', '--output', 
        help='The directory to which output files will be saved', 
        type=str, required=False
    )
    parser.add_argument(
        '-k', '--neighbors',
        help='The number of nearest neighbors to '
             'use when clustering. The default is 30.', 
             default=30, type=int, required=False
    )
    parser.add_argument(
        '-r', '--resolution',
        help='The resolution controls the coarseness of '
        'the clustering. Higher values lead to more clusters. '
        'The default is 1.0.', default=1.0, type=float, required=False
    )
    parser.add_argument(
        '-c', '--method',
        help='Include a column with the method name in the output files.',
        action="store_true", required=False
    )
    parser.add_argument(
        '-y', '--config',
        help='A yaml config file that states whether '
        'the input data should be log/logicle transformed.',
        type=str, required=False
    )
    parser.add_argument(
        '--force-transform',
        help='Log transform the input data. If omitted, '
        'and --no-transform is omitted, log transform is only '
        'performed if the max value in the input data is >1000.',
        action='store_true', required=False
    )
    parser.add_argument(
        '--no-transform',
        help='Do not perform Log transformation on the input data. '
        'If omitted, and --force-transform is omitted, log transform '
        'is only performed if the max value in the input data is >1000.',
        action='store_true', required=False
    )
    args = parser.parse_args()
    return args


def getDataName(path):
    '''
    Get input data file name
    '''
    
    # get filename from end of input path
    fileName = path.split('/')[-1]

    # get data name by removing extension from file name
    dataName = fileName[:fileName.rfind('.')] 
    return dataName


def writePatches(adata):
    '''
    Write PATCHES_FILE from leidenCluster() adata
    '''
    print("Writing Patches...")
    
    # extract patch IDs to dataframe
    patches = pd.DataFrame(adata.obs[PATCH_ID].astype(int))

    # extract and add cluster assignments to patches dataframe
    patches[CLUSTER] = adata.obs[LEIDEN] 

    # add in method column if requested
    if args.method:
        patches[METHOD] = SCANPY

    patches.to_csv(f'{output}/{patches_file}', index=False)


def writeClusters(adata):
    '''
    Write CLUSTERS_FILE from leidenCluster() adata
    '''
    print("Writing Clusters...")
    clusters = pd.DataFrame(
        columns=adata.var_names, index=adata.obs[LEIDEN].cat.categories
    )
    # name indices as cluster column
    clusters.index.name = CLUSTER
    # this assumes that LEIDEN = 'leiden' if the name is changed, 
    # replace it for 'leiden' in this line
    for cluster in adata.obs.leiden.cat.categories: 
        clusters.loc[cluster] = adata[
            adata.obs[LEIDEN].isin([cluster]), :].X.mean(0)
    
    # add in method column if requested
    if args.method:
        clusters[METHOD] = SCANPY

    clusters.to_csv(f'{output}/{clusters_file}')


def getMax(df):
    '''
    Get max value in dataframe.
    '''
    return max([n for n in df.max(axis = 0)])


def leidenCluster(input_file):
    '''
    Cluster data using the Leiden algorithm via scanpy
    '''

    print("Starting leidenCluster()...")

    if input_file.endswith(".parquet"):
        data = pd.read_parquet(input_file)
    elif input_file.endswith(".csv"):
        data = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file type: {input_file}")
    #########
    # Make sure CellID is first
    columns = list(data.columns)
    if columns.index(PATCH_ID) != 0:
        columns.insert(0, columns.pop(columns.index(PATCH_ID)))
    data = data[columns]

    # Create AnnData directly
    feature_data = data.drop(columns=[PATCH_ID])

    adata = ad.AnnData(
        X=feature_data.to_numpy(),
        obs=pd.DataFrame(
            {PATCH_ID: data[PATCH_ID].values},
            index=data[PATCH_ID].astype(str)
        ),
        var=pd.DataFrame(index=feature_data.columns.astype(str))
    )

    print("Started writing config")
    # log transform the data according to parameter. 
    # If 'auto,' transform only if the max value >1000. 
    # Don't do anything if transform == 'false'. 
    # Write transform decision to yaml file.
    if transform == 'true':
        sc.pp.log1p(adata, base=10)
        writeConfig(True)
    elif transform == 'auto' and getMax(adata.X) > 1000:
        sc.pp.log1p(adata, base=10)
        writeConfig(True)
    else:
        writeConfig(False)
    print("Finished writing config")
    
    # compute neighbors and cluster
    # compute neighbors using the first 10 principle components and the number 
    # of neighbors provided in the command line. Default is 30.
    sc.pp.neighbors(
        adata, n_neighbors=args.neighbors, n_pcs=50, use_rep='X')
    # run leidan clustering. default resolution is 1.0
    sc.tl.leiden(
        adata, key_added=LEIDEN, resolution=args.resolution, flavor="igraph", 
        n_iterations=2, directed=False) 
    print("Finished leiden clustering")
    
    # write patch/cluster information to 'PATCHES_FILE'
    writePatches(adata)

    # write cluster mean feature expression to 'CLUSTERS_FILE'
    writeClusters(adata)


def writeConfig(transformed): 
    '''
    Write to a yaml file whether the data was transformed or not.
    '''
    qcExists = os.path.exists('qc')
    if not qcExists: 
        os.mkdir('qc')
    with open('qc/config.yml', 'a') as f:
        f.write('---\n')
        if transformed:
            f.write('transform: true')
        else:
            f.write('transform: false')


def readConfig(file):
    '''
    Read config.yml file contents.
    '''
    f = open(file, 'r')
    lines = f.readlines()

    # find line with 'transform:' in it
    for line in lines:
        if 'transform:' in line.strip():
            # get last value after colon
            transform = line.split(':')[-1].strip()

    return transform


if __name__ == '__main__':
    '''
    Main.
    '''
    # parse arguments
    args = parseArgs()

    # get user-defined output dir 
    # (strip last slash if present) or set to current
    if args.output is None:
        output = '.'
    elif args.output[-1] == '/':
        output = args.output[:-1]
    else:
        output = args.output

    # assess log transform parameter
    if args.force_transform and not args.no_transform:
        transform = 'true'
    elif not args.force_transform and args.no_transform:
        transform = 'false'
    elif args.config is not None:
        transform = readConfig(args.config)
    else:
        transform = 'auto'

    # constants
    PATCH_ID = 'CellID'  # column name holding patch IDs
    CLUSTER = 'Cluster'  # column name holding cluster number
    LEIDEN = 'leiden'  # obs name for cluster assignment
    METHOD = 'Method'  # name of column containing the method for clustering
    SCANPY = 'Scanpy'  # the name of this method
    
    # output file names
    
    # get the name of the input data file to 
    # add as a prefix to the output file names
    data_prefix = getDataName(args.input)
    
    # name of output cleaned data CSV file
    clean_data_file = f'{data_prefix}-clean.csv'
    
    # name of output CSV file that contains the mean 
    # expression of each latent vector, for each cluster
    clusters_file = f'{data_prefix}-clusters.csv'

    # name of output CSV file that contains each
    # patch ID and it's assigned cluster
    patches_file = f'{data_prefix}-patches.csv' 

    print("Entering Leiden function...")

    # cluster using scanpy implementation of Leiden algorithm
    leidenCluster(args.input)
