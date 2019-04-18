from __future__ import print_function
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
import cv2
import cPickle as pickle
# from Debugging.pixel2mesh.help.initGraph_help import *


class InitGraph(object):
    def __init__(self, size=[6, 8], name='init_graph'):
        self.init_dt = np.dtype(
            [('index', np.int64, 2), ('position', np.float16, 2), ('coord', np.int16, 2)])
        self.init = np.empty(size, dtype=self.init_dt)
        self.size = np.asarray(size)
        self.lane = self.size[0]
        self.height = self.size[1]
        self.vertices = self.lane*self.height
        self.param = {'top': 0.5, 'bot': -0.5, 'up_len': 0.1, 'low_len': 1}
        self.coord = np.array([]).reshape((0, 3))
        self.support = []
        self.pool_idx = []
        self.pool_mat = None
        self.faces = None
        self.data6 = None
        self.lapn_idx = []

        # Generate Coord
        # `You Only Look Once' :) in this part
        self.class_vec = self.class_assign()
        self.coord_gen(self.height)

        # Three Blocks iteration to get some parameters
        for height, vertices, edges, block_idx in self.block_generator():
            # Generate Pooling index
            if block_idx > 1:
                self.pool_idx.append(self.pool_idx_gen(height, block_idx))
            # Generate Support recursively
            self.support.append(self.support_gen(vertices, height))
            self.lapn_idx.append(self.lapn_idx_)

        self.height_end = height

        # Generate .dat file
        self.init_graph_data = (self.coord,
                                self.support[0], self.support[1], self.support[2],
                                self.pool_idx, self.faces,
                                self.data6, self.lapn_idx)
        self.write_init_graph()

    ######################################
    # Initial Graph .dat generator block #
    ######################################

    def block_generator(self, block_num=3):
        lane = self.lane.copy()
        height = self.height.copy()
        vertices = self.vertices.copy()
        iteration = 0
        while iteration < block_num:
            edge = 2 * lane * height - lane - height
            yield height, vertices, edge, iteration+1
            height = 2 * height - 1
            vertices = height * lane
            iteration = iteration + 1


    ### Coord Functions ###

    def class_assign(self):
        half_lane = self.lane / 2
        class_vec = np.arange(1, self.lane+1, 2)
        class_vec_ = class_vec[:half_lane] + 1
        return np.concatenate((class_vec, class_vec_[::-1]))

    def coord_gen(self, height):
        # First to generate the right-most line of dots x,y coord
        # upper/lower linspace by x axis
        x_linsp = np.linspace(
            self.param['up_len'], self.param['low_len'], num=height)
        # height linspace by y axis
        y_linsp = np.linspace(self.param['top'], self.param['bot'], num=height)

        for i in range(height):
            x_coord = np.linspace(-x_linsp[i], x_linsp[i], num=self.lane)
            y_coord = np.tile(y_linsp[i], self.lane)
            coord_temp = np.vstack((x_coord, y_coord, self.class_vec))
            self.coord = np.vstack((self.coord, np.transpose(coord_temp)))


    ### Pooling_idx Functions ###

    def pool_pair(self, pool_mat):
        pool_idx = np.array([], dtype=np.int).reshape((0, 2))
        # Do not include the last line
        for sublist1, sublist2 in zip(pool_mat[:-1], pool_mat[1:]):
            for item1, item2 in zip(sublist1, sublist2):
                pool_idx = np.vstack((pool_idx, np.asarray([item1, item2])))
        return pool_idx

    def pool_idx_gen(self, height, block_idx):
        # Generate natural ordered index matrix
        nat_mat = np.arange(self.lane*height).reshape((height, self.lane))
        # Pool_mat is the same size matrix waits to be fullfilled.
        pool_mat = np.empty(shape=nat_mat.shape, dtype=np.int)
        # Marker p are sign for the starting row index
        # Marker l indicates that p+l is the ending row index for
        p = 0
        for i in range(block_idx):
            step = 2 ** min(block_idx-1, block_idx-i)
            l = len(nat_mat[i::step])
            if i == 0:
                pool_mat[0::step] = nat_mat[p:p+l]
            else:
                pool_mat[step/2::step] = nat_mat[p:p+l]
            p = p + l
        self.pool_mat = pool_mat.copy()
        return self.pool_pair(pool_mat)


    ### Support Functions ###

    def pair_num_gen(self, list_array, default_value=[2, 3, 4]):
        dicts = []
        # To generate a dictionary, key as the 'natural' index, and value as number of connected nodes
        for list_, val in zip(list_array, default_value):
            dicts.append(dict((key, val) for key in list_.reshape(-1)))
        pair_num_dict = {}
        for d in dicts:
            for k, v in d.iteritems():
                pair_num_dict[k] = v
        return pair_num_dict

    def support_gen_identity(self, vertices):
        # Generate Identity Sparse Matrix
        indices = np.asarray([[x, x] for x in range(vertices)])
        value = np.ones([vertices, ])
        shape = np.array([vertices, vertices])
        return (indices, value, shape)

    def support_gen_adjacency(self, vertices, height):
        indices = np.array([], dtype=np.int32).reshape((0, 2))
        value = []
        shape = np.array([vertices, vertices])

        # initialize the laplacian term!
        self.lapn_idx_ = np.ones((vertices, 10), dtype=np.int)*-1
        self.lapn_idx_[:, -2] = np.arange(vertices)

        # Classify all the indices into three categories
        # corresponding to the corners, edges and inner indices
        # Use roll to recursively assign pair indices
        # Pair_movement is clockwise, starting from 3 o'clock
        pair_2_index_list = [0, self.lane-1, vertices-1, vertices-self.lane]
        pair_3_index_list = [range(1, self.lane-1),
                             range(2*self.lane-1, vertices-1, self.lane),
                             range(vertices-self.lane+1, vertices-1),
                             range(self.lane, vertices-self.lane, self.lane)]
        pair_3_index_list_ = [
            val for sublist in pair_3_index_list for val in sublist]
        pair_4_index_list = list(
            set(range(vertices)) - set(pair_3_index_list_) - set(pair_2_index_list))
        
        # Natural indexed matrix
        pair_2_index_list = np.array(pair_2_index_list)
        pair_3_index_list = np.array(pair_3_index_list)
        pair_4_index_list = np.array(pair_4_index_list)
        pair_3_index_list_ = np.array(pair_3_index_list_)

        # If in block 1, there's no pooling so the pooling mat equals the natural 
        # indexed matrix
        if self.pool_mat is None:
            self.pool_mat = np.arange(vertices).reshape((height, self.lane))

        # Map these lists as dictionary keys to corresponding linked node numbers as values
        # Note that these keys are NATURAL indices of entries,
        # not the non-natural pooling indices!!!
        pair_num_dict = self.pair_num_gen(
            [pair_2_index_list, pair_3_index_list_, pair_4_index_list])

        # Use non-zero to indicate that movement needs to be carried out
        # While zero entry mean that is an invalid movement
        pair_movement = np.array([1, self.lane, -1, -self.lane])
        pair_2_move_indicator = np.array([1, 1, 0, 0])
        pair_3_move_indicator = np.array([1, 1, 1, 0])

        # Pair_2
        for idx in pair_2_index_list:
            pair_2_move = pair_movement*pair_2_move_indicator
            # initialize lapn_temp, to store the nodes of a vertex
            lapn_temp = []
            for move in pair_2_move:
                if move != 0:
                    # get the non-natural real pooling idx from natural entry idx
                    index_plus = np.take(self.pool_mat, [idx, idx + move])
                    indices = np.vstack((indices, index_plus))
                    value.append(
                        1./np.sqrt(pair_num_dict[idx]/np.sqrt(pair_num_dict[idx+move])))
                    # store in lapn_temp
                    lapn_temp.append(index_plus[1])
            self.lapn_idx_gen(index_plus[0], lapn_temp, pair_num_dict[idx])
            pair_2_move_indicator = np.roll(pair_2_move_indicator, 1)

        # Pair_3
        for sublist in pair_3_index_list:
            pair_3_move = pair_movement*pair_3_move_indicator
            for idx in sublist:
                lapn_temp = []
                for move in pair_3_move:
                    if move != 0:
                        index_plus = np.take(self.pool_mat, [idx, idx + move])
                        indices = np.vstack((indices, index_plus))
                        value.append(
                            1./np.sqrt(pair_num_dict[idx]/np.sqrt(pair_num_dict[idx+move])))
                        lapn_temp.append(index_plus[1])
                self.lapn_idx_gen(index_plus[0], lapn_temp, pair_num_dict[idx])
            pair_3_move_indicator = np.roll(pair_3_move_indicator, 1)

        # Pair_4
        for idx in pair_4_index_list:
            lapn_temp = []
            for move in pair_movement:
                index_plus = np.take(self.pool_mat, [idx, idx + move])
                indices = np.vstack((indices, index_plus))
                value.append(
                    1./np.sqrt(pair_num_dict[idx]/np.sqrt(pair_num_dict[idx+move])))
                lapn_temp.append(index_plus[1])
            self.lapn_idx_gen(index_plus[0], lapn_temp, pair_num_dict[idx])
        return (indices, value, shape)

    def support_gen(self, vertices, height):
        identity_sp = self.support_gen_identity(vertices)
        adjacency_sp = self.support_gen_adjacency(vertices, height)
        return [identity_sp, adjacency_sp]


    ### Generate Faces idx: which does not exist ;) ###

    def faces_gen(self):
        pass


    ### Generate Laplacian idx ###

    def lapn_idx_gen(self, vertex_idx, lapn_temp, pair_num):
        assert pair_num==len(lapn_temp)
        self.lapn_idx_[vertex_idx,:pair_num] = np.asarray(lapn_temp)
        self.lapn_idx_[vertex_idx,-1] = pair_num
        pass

    def write_init_graph(self, write_path='./Debugging/pixel2mesh/help/initGraph.dat', remove_origin=True):
        if remove_origin:
            try:
                os.remove(write_path)
            except OSError:
                pass
        # Write the new .dat file of Initial Graph
        with open(write_path, 'wb') as data_file:
            print('Writing to ' + write_path)
            pickle.dump(self.init_graph_data, data_file)
            print('Done')

    # Close this object, to generate .dat file with multiple images
    def close(self):
        pass

    #########################################
    # Training Dataset .dat generator block #
    #########################################
    def mid_lane_det(self):
        pass

    def __call__(self, name, lanes, h_samples):
        self.data_dt = np.dtype(
            [('', np.int64, 2), ('position', np.float16, 2), ('coord', np.int16, 2)])

    # Release discarded memories

    def __del__(self):
        pass

    # Write in new entry of image information
    def write_img(self):
        # pickel IO support in here!!!
        pass


if __name__ == "__main__":
    initGraph = InitGraph()
    pass
