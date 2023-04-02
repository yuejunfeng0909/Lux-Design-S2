from gym import spaces

light_robot_action_space = spaces.Dict({
    # action_type: move, transfer resource, pickup power, dig, self destruct, no op
    "action_type": spaces.Discrete(6),
    
    # direction: center, up, right, down, left
    "direction": spaces.Discrete(5),
    
    # resource_type: ice, ore, water, metal, power
    "resource_type": spaces.Discrete(5),
    
    # percentage of resource to transfer
    # TODO is there a better implementation?       
    "resource_amount": spaces.Box(0, 1),
})

# heavy robot
heavy_robot_action_space = spaces.Dict({
    # move, transfer resource, pickup power, dig, self destruct, no op
    "action_type": spaces.Discrete(6),
    
    # center, up, right, down, left
    "direction": spaces.Discrete(5),
    
    # ice, ore, water, metal, power
    "resource_type": spaces.Discrete(5), 
})

# factory
factory_action_space = spaces.Dict({
    # build light robot, build heavy robot, water, no op
    "action_type": spaces.Discrete(4),
})

# assemble action space
action_space = spaces.Dict({
    "light_robot": light_robot_action_space,
    "heavy_robot": heavy_robot_action_space,
    "factory": factory_action_space
})

print(action_space.sample()["light_robot"]["resource_amount"])
