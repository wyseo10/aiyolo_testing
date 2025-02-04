from crazyflie_py import Crazyswarm

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0

def main():
    print("Waiting for the cfserver (ros2 launch crazyflie launch.py)")
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    print("Connected to the cfserver")

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)

if __name__ == '__main__':
    main()