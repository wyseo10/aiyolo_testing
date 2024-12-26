#!/usr/bin/env python3
#hovering test

from crazyflie_py import Crazyswarm
import numpy as np

def hovering():
    Z = 0.3
    
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs


    cf.takeoff(targetHeight=Z, duration=1.0+Z)
    timeHelper.sleep(1.5+Z)
    for cf in allcfs.crazyflies:
        pos = np.array(cf.initalPosition) + np.array([0, 0, Z])
        cf.goTo(pos, 0, 1.0)
    