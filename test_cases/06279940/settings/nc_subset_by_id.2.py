#!/usr/bin/env python
# Script to subset a netcdf file based on a list of HUC ids.
# hardwired for hruIds at moment; Doesn't work with gru ids in index at present
# B. Nijssen script
# AWW-June2020 modified to allow option to write output in order of ID list or order
#              of input netcdf file

import argparse
from datetime import datetime
import os
import sys
import xarray as xr
import numpy as np

def process_command_line():
    '''Parse the commandline'''
    parser = argparse.ArgumentParser(description='Script to subset a netcdf file based on a list of IDs.')
    parser.add_argument('ncfile',
                        help='path of netcdf file that will be subset.')
    parser.add_argument('idfile', 
                        help='path of file with list of ids.')
    #parser.add_argument('opath',
    #                    help='directory where subsetted file will be written.')
    parser.add_argument('ofile',
                        help='output file (netcdf)')
    parser.add_argument('out_order',
                        help='desired order of output. 0:use id list order; 1:use input nc file order')
    args = parser.parse_args()
    return(args)



# main
if __name__ == '__main__':
    # process command line
    args = process_command_line()

    # read  the IDs to subset
#    with open(args.idfile) as f:
#        ids = [int(x.strip()) for x in f if x.strip()]

    # ingest the target netcdf file with an id list (eg the attributes file of target domain)
    ds_targ_ids = xr.open_dataset(args.idfile, decode_times=False)
    ids = ds_targ_ids.hruId.values
    
    # ingest the netcdf file
    ds = xr.open_dataset(args.ncfile, decode_times=False)

    # subset the netcdf file based on the hruId
    ds_subset = ds.where(ds.hruId.isin(ids), drop=True)

    # reorder to match input id list (not order from full domain shapefile)
    if int(args.out_order) == 0:
        print('writing %s in id list order' % args.ofile)
        remap_idx = np.where(ds_subset.hruId.values[None, :] == np.array(ids)[:, None])[1]
        ds_subset = ds_subset.isel(dict(hru=remap_idx))
    else:
        print('writing %s in original domain order' % args.ofile)

    # Write to file
    ds_subset.to_netcdf(args.ofile)

    # Write IDs from the ID file that were not in the NetCDF file to stdout
    missing = set(ids).difference(set(ds_subset.hruId.values))
    if missing:
        print("Missing IDs: ")
        for x in missing:
            print(x)

            
            
            
            
# other stuff that could be used    

# update the history attribute (or not)
    #history = '{}: {}\n'.format(datetime.now().strftime('%c'), 
    #                            ' '.join(sys.argv))
    #if 'history' in ds_subset.attrs:
    #    ds_subset.attrs['history'] = history + ds_subset.attrs['history']
    #else:
    #    ds_subset.attrs['history'] = history

#ofile = os.path.join(args.opath, os.path.basename(args.ncfile))
    #ofile = args.opath
