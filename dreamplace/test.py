##
# @file   test.py
# @author NC Lin
# @date  
# @brief  Test read file and etc.
#
"""
import sys

sys.path=['/DREAMPlace/install/dreamplace', '/opt/conda/lib/python39.zip', '/opt/conda/lib/python3.9', '/opt/conda/lib/python3.9/lib-dynload', '/opt/conda/lib/python3.9/site-packages', '/DREAMPlace/install']
"""

import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import numpy as np
import logging
import math
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace
import pdb

def make_database(filename):
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                            format='[%(levelname)-7s] %(name)s - %(message)s',
                            stream=sys.stdout)

    params = Params.Params()

    params.load(filename)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    return placedb
    # print(placedb.row_height)

    # for i in range(0,placedb.num_physical_nodes-placedb.num_terminals-placedb.num_terminal_NIs):
    #     id = placedb.node_name2id_map[placedb.node_names[i].decode("utf-8") ]
    #     if placedb.node_size_y[id]>placedb.row_height and placedb.node_size_x[id]>placedb.row_height:
    #         print(placedb.node_names[i].decode("utf-8"),placedb.node_x[id],placedb.node_y[id])

    #a,b=enum_macro_cell(placedb)

    #print("num of movable macro: ",str(len(a)))
    #print("num of movable std cell: ",str(len(b)))

    # placedb.read_pl(params,"/DREAMPlace/benchmarks/PA5590/PA5590.pl")

    # for i in range(0,placedb.num_physical_nodes-placedb.num_terminals-placedb.num_terminal_NIs):
    #     id = placedb.node_name2id_map[placedb.node_names[i].decode("utf-8") ]
    #     if placedb.node_size_y[id]>placedb.row_height and placedb.node_size_x[id]>placedb.row_height:
    #         print(placedb.node_names[i].decode("utf-8"),placedb.node_x[id],placedb.node_y[id])

    #print(sys.path)



# dont use
def enum_movable_macro_cell(placedb):
    macro_ids = []
    std_ids=[]

    for i in range(0, placedb.num_physical_nodes-placedb.num_terminals-placedb.num_terminal_NIs):
        id=placedb.node_name2id_map[placedb.node_names[i].decode("utf-8")]
        if placedb.node_size_y[id]>placedb.row_height and placedb.node_size_x[id]>placedb.row_height:
            macro_ids.append(id)
        else:
            std_ids.append(id)
    return macro_ids,std_ids



def overlap(xl1,xh1,yl1,yh1,xl2,xh2,yl2,yh2):
    """
return overlap between two rec.
    """
    return max(min(xh1, xh2)-max(xl1, xl2), 0.0) * max(min(yh1, yh2)-max(yl1, yl2), 0.0)

def construct_density_map(size, pldb):
    dmap=np.zeros((size,size),dtype=np.float32)
    chip_x=pldb.xh-pldb.xl
    chip_y=pldb.yh-pldb.yl
    bin_size=np.ceil(max(chip_x, chip_y)/size)


    # fill outside
    if chip_x>chip_y:
        # fill x=0~~chip_x y=chip_y~~chip_x
        dmap[0:size, int(np.ceil(chip_y/bin_size)):size] = 1
        if np.ceil(chip_y/bin_size) != chip_y/bin_size:
            dmap[0:size, int(np.ceil(chip_y/bin_size))-1] = \
                np.ceil(chip_y/bin_size) - chip_y/bin_size
    if chip_y>chip_x:
        # fill x=chip_x~~chipy y=0~~chip_y
        dmap[int(np.ceil(chip_x/bin_size)):size, 0:size]=1
        if np.ceil(chip_x/bin_size) != chip_x/bin_size:
            dmap[int(np.ceil(chip_x/bin_size))-1, 0:size] = \
                np.ceil(chip_x/bin_size) - chip_x/bin_size
    bin_index_xl = np.maximum(np.floor(pldb.node_x/bin_size).astype(np.int32), 0)
    bin_index_xh = np.minimum(np.ceil((pldb.node_x+pldb.node_size_x[:len(pldb.node_x)])/bin_size).astype(np.int32), \
        size-1)

    bin_index_yl = np.maximum(np.floor(pldb.node_y/bin_size).astype(np.int32), 0)
    bin_index_yh = np.minimum(np.ceil((pldb.node_y+pldb.node_size_y[:len(pldb.node_y)])/bin_size).astype(np.int32), \
        size-1)


    for i in range(pldb.num_movable_nodes,pldb.num_movable_nodes+pldb.num_terminals):
        for ix in range(bin_index_xl[i],bin_index_xh[i]+1):
            for iy in range(bin_index_yl[i],bin_index_yh[i]+1):
                dmap[ix,iy]+=overlap(ix,ix+1,iy,iy+1,\
                    pldb.node_x[i]/bin_size, (pldb.node_x[i]+pldb.node_size_x[i])/bin_size, pldb.node_y[i]/bin_size,(pldb.node_y[i]+pldb.node_size_y[i])/bin_size)

    dmap = np.minimum(dmap, 1.)
    return dmap

def add_macro2map(dmap, macro_id, pldb):
    chip_x=pldb.xh-pldb.xl
    chip_y=pldb.yh-pldb.yl
    bin_size=np.ceil(max(chip_x, chip_y)/len(dmap))

    bin_index_xl = np.maximum(np.floor(pldb.node_x[macro_id]/bin_size).astype(np.int32), 0)
    bin_index_xh = np.minimum(np.ceil((pldb.node_x[macro_id]+pldb.node_size_x[macro_id])/bin_size).astype(np.int32), \
        len(dmap)-1)

    bin_index_yl = np.maximum(np.floor(pldb.node_y[macro_id]/bin_size).astype(np.int32), 0)
    bin_index_yh = np.minimum(np.ceil((pldb.node_y[macro_id]+pldb.node_size_y[macro_id])/bin_size).astype(np.int32), \
        len(dmap)-1)


    return dmap

def compute_bin_size(pldb, method='polar') -> float:
    macros,cells=enum_macro_cell(pldb)
    avg_macro_size=sum([pldb.node_area[i] for i macros]) / len(macros)

    print("avg_macro_size is ",str(avg_macro_size))

    if method=='polar':
        # 
        return math.ceil( math.sqrt(1*avg_macro_size/0.6) )


