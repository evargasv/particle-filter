#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from functions import angle_wrap

#===============================================================================
class ParticleFilter(object):
    '''
    Class to hold the whole particle filter.

    p_wei: weights of particles in array of shape (N,)
    p_ang: angle in radians of each particle with respect of world axis, shape (N,)
    p_xy : position in the world frame of the particles, shape (2,N)
    '''

    #===========================================================================
    def __init__(self, room_map, num, odom_lin_sigma, odom_ang_sigma,
                 meas_rng_noise, meas_ang_noise):
        '''
        Initializes the particle filter
        room_map : an array of lines in the form [x1 y1 x2 y2]
        num      : number of particles to use
        odom_lin_sigma: odometry linear noise
        odom_ang_sigma: odometry angular noise
        meas_rng_noise: measurement linear noise
        meas_ang_noise: measurement angular noise
        '''

        # Copy parameters
        self.map = room_map
        self.num = num
        self.odom_lin_sigma = odom_lin_sigma
        self.odom_ang_sigma = odom_ang_sigma
        self.meas_rng_noise = meas_rng_noise
        self.meas_ang_noise = meas_ang_noise

        # Map
        map_xmin = np.min(self.map[:, 0])
        map_xmax = np.max(self.map[:, 0])
        map_ymin = np.min(self.map[:, 1])
        map_ymax = np.max(self.map[:, 1])

        # Particle initialization
        self.p_wei = 1.0 / num * np.ones(num)
        self.p_ang = 2 * np.pi * np.random.rand(num)
        self.p_xy  = np.vstack(( map_xmin + (map_xmax - map_xmin) * np.random.rand(num),
                                 map_ymin + (map_ymax - map_ymin) * np.random.rand(num) ))

    #===========================================================================
    def predict(self, odom):
        '''
        Moves particles with the given odometry.
        odom: incremental odometry [delta_x delta_y delta_yaw] in the vehicle frame
        '''

        # Add Gaussian noise to odometry measures
        lin_noise = np.random.randn(self.num,2) * self.odom_lin_sigma
        ang_noise = angle_wrap( np.random.randn(self.num) * self.odom_ang_sigma + odom[2] )

        # Increment particle positions in correct frame
        odom_noise = np.tile(odom[0:2], (self.num, 1))
        odom_noise += lin_noise

        self.p_xy[0,:] += odom_noise[:,0] * np.cos(self.p_ang) - odom_noise[:,1] * np.sin(self.p_ang)
        self.p_xy[1,:] += odom_noise[:,0] * np.sin(self.p_ang) + odom_noise[:,1] * np.cos(self.p_ang)

        # Increment angley
        self.p_ang += ang_noise
        self.p_ang = angle_wrap(self.p_ang)

    #===========================================================================
    def weight(self, lines):
        '''
        Look for the lines seen from the robot and compare them to the given map.
        Lines expressed as [x1 y1 x2 y2].
        '''

        # Constant values for all weightings
        val_rng = 1.0 / (self.meas_rng_noise * np.sqrt(2 * np.pi))
        val_ang = 1.0 / (self.meas_ang_noise * np.sqrt(2 * np.pi))

        # origin of the map
        orig = np.array( [0, 0, 0] )

        # map and measured lines array for polar coordinates
        expected_lines = np.zeros( (self.map.shape[0], 2) )
        measured_lines = np.zeros( (lines.shape[0], 2) )

        # Loop over particles
        for i in range(self.num):

            # Position of the particle
            odom = np.array( [self.p_xy[0,i], self.p_xy[1,i], self.p_ang[i] ] )
            # Weight of each measured line
            lines_wei = np.zeros( lines.shape[0] )

            # Transform map lines to local frame and to [range theta]
            for j in range(self.map.shape[0]):
                # Transform to [range theta]
                expected_lines[j,:] = self.get_polar_line( self.map[j,:], odom )

            # Transform measured lines to [range theta] and weight them
            for j in range(lines.shape[0]):

                # Transform measured lines
                measured_lines[j,:] = self.get_polar_line( lines[j,:], orig  )
                map_wei = np.zeros( expected_lines.shape[0] )

                # Select the map line that best corresponds to the measured line. The weight of them is calculated and the maximum is selected
                for k in range(expected_lines.shape[0]):

                    # Weight them
                    range_wei = val_rng * np.exp((-( measured_lines[j,0] - expected_lines[k,0] )**2.0)/(2.0*(val_rng**2.0)))
                    angle_wei = val_ang * np.exp((-( measured_lines[j,1] - expected_lines[k,1] )**2.0)/(2.0*(val_ang**2.0)))
                    map_wei[k] = range_wei * angle_wei

                # Best associated line on the map
                lines_wei[j] = map_wei.max()

            # OPTIONAL question
            # make sure segments correspond, if not put weight to zero
            #
            #

            # Take best weighting (best associated lines)
            self.p_wei[i] *= np.sum( lines_wei )

        # Normalize weights
        self.p_wei /= np.sum(self.p_wei)


    #===========================================================================

    def resample(self):
        '''
        Systematic resampling of the particles.
        '''
        # Look for particles to replicate
        p_idx = np.zeros(self.num, dtype=np.uint8)
        r = np.random.rand() * (1.0/self.num)
        c = self.p_wei[0]
        j = 0

        for i in range(self.num):
            u = r + ( i * (1.0/self.num) )
            while u > c:
                j = (j + 1) % self.num
                c = c + self.p_wei[j]
            p_idx[i] = int(j);

        # Pick chosen particles
        self.p_ang = self.p_ang[(p_idx)]
        self.p_wei = self.p_wei[(p_idx)]
        self.p_xy = self.p_xy[:,p_idx]
    #===========================================================================
    def get_mean_particle(self):
        '''
        Gets mean particle.
        '''
        # Weighted mean
        weig = np.vstack((self.p_wei, self.p_wei))
        mean = np.sum(self.p_xy * weig, axis=1) / np.sum(self.p_wei)

        ang = np.arctan2( np.sum(self.p_wei * np.sin(self.p_ang)) / np.sum(self.p_wei),
                          np.sum(self.p_wei * np.cos(self.p_ang)) / np.sum(self.p_wei) )

        return np.array([mean[0], mean[1], ang])

    #===========================================================================
    def get_polar_line(self, line, odom):
        '''
        Transforms a line from [x1 y1 x2 y2] from the world frame to the
        vehicle frame using odomotrey [x y ang].
        Returns [range theta]
        '''
        # Line points
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        # Compute line (a, b, c) and range
        line = np.array([y1-y2, x2-x1, x1*y2-x2*y1])
        pt = np.array([odom[0], odom[1], 1])
        dist = np.dot(pt, line) / np.linalg.norm(line[:2])

        # Compute angle
        if dist > 0:
            ang = np.arctan2(line[1], line[0])
        else:
            ang = np.arctan2(-line[1], -line[0])

        # Return in the vehicle frame
        return np.array([np.abs(dist), angle_wrap(ang - odom[2])])
