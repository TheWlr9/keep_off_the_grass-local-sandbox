==== Pre-requisites for running the jar file ====
	1. Java runtime environment (https://www.java.com/en/download/)
	2. Having java.exe in PATH variable

==== Command to run the simulator locally ====
	> java -jar [-Dnotimeout="true"] .\ea-2022-keep-off-the-grass-1.0-SNAPSHOT.jar <"python <path_to_py_file_for_agent1>"|path_to_exe_file_for_agent1> <"python <path_to_py_file_for_agent2>"|path_to_exe_file_for_agent2> <server|local> [<seed>]
	
	where
	1. -Dnotimeout - an optional argument, that is "False" by default. If set to "True", it stops the java code to throw an Exception when the agent's code is hanging for some reason
	   Should be activated if one wants to attach with a debugger to the agent's code
	2. path_to_py_file_for_agent1 and path_to_py_file_for_agent2 - Python files of the agents that you want to compete with each other
	3. path_to_exe_file_for_agent1 and path_to_exe_file_for_agent2 - executable files of the agents that you want to compete with each other
	4. server - if you want to run a server where you will see the simulation of the game (exactly like in the Codingame platform). Accessible at http://localhost:8888/test.html 
	   local - if you want to run the game headless and generate all the info (game state, moves, etc), in the same format as CG platform uses (the format is documented here [1])
	5. seed - optional argument, will be random generated if not provided. Represents the seed of the game.

[1] https://codingame.github.io/codingame-game-engine/com/codingame/gameengine/runner/simulate/GameResult.html


Example: 
java --add-opens java.base/java.lang=ALL-UNNAMED -Dnotimeout="true" -jar ea-2022-keep-off-the-grass-1.0-SNAPSHOT.jar "python ai.py -f _new_v1" "python Boss-easy.pyc" local

Output:
http://localhost:8888/test.html
Exposed web server dir: C:\Users\[name]\AppData\Local\Temp\codingame