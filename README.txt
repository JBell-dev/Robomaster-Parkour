#Note:

The main file to run is maincontrol.py
the other two files are the construction parts for our final maincontrol implementation. 

-> ros2 run rm_parkour maincontrol 

in addition, the bridge should be run on parallel: 

-> ros2 launch robomaster_ros main.launch device:="tcp:host=localhost;port=33333" simulation:=True name:=RoboMaster

#Video:

We added in the video folder the final main video of the implementation. In addition, here you can find extra videos of it:

* https://www.youtube.com/watch?v=V_oNyCZTsEY
* https://www.youtube.com/watch?v=0bVYYBI61h0

#Extra:

we added a .txt with the lua code that we could achieve to expose the topics of a quadcopter following the guided steps on the forum : https://manual.coppeliarobotics.com/en/ros2Tutorial.htm , however, we could not make the action get carry out by the simulator 
given an error with a function that is not in the scrypt despite of the fact that seems to be receiving the commands when we echo the topics. 

Furthermore, is also possible to find "get_largest" that is an approximation algorithm we designed as mentioned in section B of the report "identifying the gates"