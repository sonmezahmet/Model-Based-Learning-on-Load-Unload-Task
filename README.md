# Model Based Learning on Load Unload Task (Prioritized Sweeping)
## Problem Definition
In this project, the problem is implementing the prioritized sweeping algorithm which
makes model based learning. Also environment should be created dynamically according
to user’s input.
<br>
## Methodology
### Creating Environment
![image](https://user-images.githubusercontent.com/56430166/187248501-3805ab43-63ac-410a-bb9a-1bf5c571791c.png)
<br>
Before I started creating the environment, I divided the environment into different states. I
created 9 different classes for those states which are, ‘State’, ‘RegularState’, LoadState’,
‘UnloadState’, ‘RoughState’, ‘BlockState’, ’SlopeState’, ‘UpperSlope’ and ‘LowerSlope’.
‘State’ is base class for others. ‘RegularState’ presents normal states which can be
moveable. ‘RoughState’ presents rough roads on the grid world. ‘BlockState’ presents
can’t be moveable states. ‘SlopeState’ indicates that state is slope state, and it contains
2 other sub states which are ‘UpperSlope’ and ‘LowerSlope’. Each state has 3 different Q
tables and 3 different ET table. First Q table and ET table is used when agent isn’t loaded
yet, second Q table and ET table is used when agent is loaded and isn’t unloaded yet,
and last Q table and ET table is used when agent is unloaded.
Environment class has only one class parameter which is ’n’ for size of grid world. For ’n’,
minimum value is 6, e.g., if you try create 5x5 environment, it creates automatically 6x6
environment.
<br>
Environment class has 3 class functions which are ‘__create_2d_matrix' for creating n x n
matrix, ‘__create_environment’ for placing proper states to created n x n matrix,
‘print_environment’ for printing environment.
### Implementing Prioritized Sweeping
The algorithm of prioritized sweeping is depicted in below figure. <br>
![image](https://user-images.githubusercontent.com/56430166/187256808-f675edeb-9d0e-4e3d-87d8-e8d54588d959.png) <br>
For implementing model, I created model dictionary and after each observation I added
new list to this dictionary with a key which represents current state. It is depicted in below figure. <br>
![image](https://user-images.githubusercontent.com/56430166/187256953-9d797706-b4db-4e6d-b651-af9de735f393.png) <br>
I calculated ‘p’ value and if this value is greater than given threshold, then I added this
state to the priority queue with priority p. It is depicted in below figure. <br>
![image](https://user-images.githubusercontent.com/56430166/187257123-f3d8ff22-bf1f-4898-aa43-1a0da55c4ad9.png) <br>
In loop (which runs n times), I calculated TD error and according to that, I updated
appropriate Q table. It is depicted in below figure. <br>
![image](https://user-images.githubusercontent.com/56430166/187257272-3b2b8cc8-8429-44df-aa92-b4da525533c5.png) <br>
Finally, I find all s”, a” pairs which lead to s from previously experienced pairs (model).
And according to algorithm I calculated ‘p’ value. If ‘p’ value is greater than given
threshold value, then I added this s”, a” pair to priority queue. It is depicted in below figure. <br>
![image](https://user-images.githubusercontent.com/56430166/187257403-dbd93b32-c302-493c-a0c6-d820a12a90b5.png) <br>
## Experiments and Results
I ran this algorithm for 100 episodes with learning rate is 0.3, discount rate is 0.9, epsilon
is 0.1, threshold is 0.01, and n is 4 in 10x10 environment. Results shown in figure. <br>
![image](https://user-images.githubusercontent.com/56430166/187257881-2f774b48-1813-4d35-becd-d77c04b875a7.png) <br>
## Conclusion
As seen in the experiment, prioritized sweeping algorithm only needs few example to
converge optimum policy. This makes this algorithm strong while there are few examples
from environment.






