//
//  application.h
//  main
//
//  Created by Daniel Cho on 5/30/26.
//

#pragma once

#include <vector>

#include "engine/common/system.h"
#include "engine/graphics/graphics.h"

class Application {
    
private:
    
    Graphics* graphics;
    std::vector<System*> systems;
    
public:
    void init();
    void start();
    void cleanup();
    
};
