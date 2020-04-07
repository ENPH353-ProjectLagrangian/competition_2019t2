# 2019 Term 2 UBC Parking competition

This is a simulation environment designed to match the real competition environment. You do not need to use it, though it might be helpful to test ideas and train control models.

## Installation instructions:
* If you **do not** have a catkin workspace, please create one. You can use any folder to create it in. The example below uses your home directory.
```
mkdir -p ~/353_ws/src
```

* Clone the repository into the catkin workspace src folder.
```
cd ~/353_ws/src
git clone https://github.com/ENPH353-ProjectLagrangian/competition_2019t2.git
```

* Build the packages
```
cd ~/353_ws
catkin_make
```

* Source the environment
```
source devel/setup.bash
```

* Start the simulated world
```
cd ~/353_ws/src/competition_2019t2/scripts
./run_sim.sh -g
```
You can run the simulation with or without any options. So far there is one option available:

| Option | Description      |
|:-------|:---------------- |
| -g     | generate new random license plates |

## Working within the simulation:
Once the simulation is started you can interact with it in multiple ways.

* If you want to control the robot you can use teleop_twist_keyboard:
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
```

* If you would like to watch the camera feed you can use rqt_image_view:
```
rosrun rqt_image_view rqt_image_view 
```

* The robot parameters are stored in:
```
competition_2019t2/urdf/robot.xacro
```
we use the same plugins we used in Lab 03.

# RESOURCES:
https://blog.victormeunier.com/posts/keras_multithread/