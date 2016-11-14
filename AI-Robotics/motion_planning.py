# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:02:00 2016

@author: s6324900
"""

# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]

# if heuristic is set to all 0, it resorts to classic search
heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]
#print(len(grid))
#5
#print(len(grid[0]))
#6
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost,heuristic):
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    # expand keeps track of what sequence was the search expanded
    # initialize with -1 
    expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    action = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    policy = [['' for col in range(len(grid[0]))] for row in range(len(grid))]
    seq = 0
    closed[init[0]][init[1]] = 1
    x = init[0]
    y = init[1]
    g = 0
    h = heuristic[x][y]
    f = g + h
    open_list = [[f,g,h,x,y]]
    
    found = False # flag that is set when search is complete
    resign = False #flag set if we can't find expand
    
    #print 'init open list'
    #for i in range(len(open)):
        #print('  ', open[i])
    
    while found is False and resign is False:
        # check if we still have values in open list
        if len(open_list) == 0:
            resign = True
            print('fail')
        else:
            #remove node from list
            open_list.sort(key=lambda x: x[0]) # Sort by first item 
            open_list.reverse() # Sort so it is descending 
            next = open_list.pop() #removes and returns the last item in the list
            # print 'take list item'
            # print next
            g = next[1]
            x = next[3]
            y = next[4]
            expand[x][y] = seq
            seq +=1
            
            # Check is we are done
            if x==goal[0] and y== goal[1]:
                found = True
                print(next)
                print('Succesful search')
            else:
                # expand this position and add to new open list
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                ## check if this is a valid positin
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0: # Not in closed list and not an obstacle
                            g2 = g + cost
                            h2 = heuristic[x2][y2]
                            f2 = g2 + h2
                            open_list.append([f2, g2, h2, x2, y2])
                            #seq = seq+1
                            #expand[x2][y2] = seq
                            closed[x2][y2] = 1 # Close this item for future
                            action[x2][y2] = i # action to get the state
    #Print expand
    for i in range(len(expand)):
        print(expand[i])
       
    ## Print policy
    x = goal[0]
    y = goal[1]
    policy[x][y] = '*'
    while x!= init[0] or y!= init[1]:
        x2 = x - delta[action[x][y]][0]
        y2 = y - delta[action[x][y]][1]
        policy[x2][y2] = delta_name[action[x][y]]
        x = x2
        y = y2
    for i in range(len(policy)):
        print(policy[i])
       
print search(grid,init,goal,cost, heuristic)

## Dynamic Programming - Find optimal policy for any location on the grid
def compute_value(grid,goal,cost):
    #Initialize value to 99
    value = [[99 for col in range(len(grid[0]))] for row in range(len(grid))] 
    policy = [['' for col in range(len(grid[0]))] for row in range(len(grid))]    
    change = True
    while change:
        change = False
        
        for x in range(len(grid)):
            for y in range(len(grid[1])):
                
                if goal[0] ==x and goal[1] == y:
                    policy[x][y] = '*'
                    if value[x][y] >0:
                        value[x][y] = 0
                        change = True
                        
                elif grid[x][y] == 0:
                    for i in range(len(delta)):
                        x2 = x + delta[i][0]
                        y2 = y + delta[i][1]
                        if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                            v2 = value[x2][y2] + cost
                                
                            if v2 < value[x][y]:
                                policy[x][y] = delta_name[i]
                                change = True
                                value[x][y] = v2
            
    for i in range(len(value)):
        print(value[i])
        
    for i in range(len(policy)):
        print(policy[i])
        
compute_value(grid,goal,cost)


## Actual programming assignment

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]

cost = [2, 1, 20] # cost has 3 values, corresponding to making 
                  # a right turn, no turn, and a left turn


def optimum_policy2D(grid,init,goal,cost):
    value = [[[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))]]
             
    policy = [[['' for col in range(len(grid[0]))] for row in range(len(grid))],
             [['' for col in range(len(grid[0]))] for row in range(len(grid))],
             [['' for col in range(len(grid[0]))] for row in range(len(grid))],
             [['' for col in range(len(grid[0]))] for row in range(len(grid))]]
             
    policy2D = [['' for col in range(len(grid[0]))] for row in range(len(grid))]
    change = True
    while change:
        change = False
        
        for x in range(len(grid)):
            for y in range(len(grid[1])):
                for orientation in range(4):
                
                    if goal[0] ==x and goal[1] == y:
                        if value[orientation][x][y] >0:   
                            change = True
                            value[orientation][x][y] = 0
                            policy[orientation][x][y] = '*'
                        
                    elif grid[x][y] == 0:
                        # calculate 3 different ways to propogate
                        for i in range(3):
                            o2 = (orientation + action[i])%4
                            x2 = x + forward[o2][0]
                            y2 = y + forward[o2][1]
                            if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                                v2 = value[o2][x2][y2] + cost[i]
                                    
                                if v2 < value[orientation][x][y]:
                                    value[orientation][x][y] = v2
                                    policy[orientation][x][y] = action_name[i]
                                    change = True
                                   
    x = init[0]
    y = init[1]
    orientation = init[2]
    
    policy2D[x][y] = policy[orientation][x][y]
    while  policy[orientation][x][y] != '*':
        if  policy[orientation][x][y] == '#':
            o2 = orientation
        elif  policy[orientation][x][y] == 'R':
            o2 = (orientation -1) %4
        elif  policy[orientation][x][y] == 'L':
            o2 = (orientation +1) %4
        x = x + forward[o2][0]
        y = y + forward[o2][1]
        orientation = o2
        policy2D[x][y] = policy[orientation][x][y]
    
    for i in range(len(policy2D)):
        print(policy2D[i])
        

optimum_policy2D(grid,init,goal,cost)
