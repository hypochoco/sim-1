//
//  render_system.h
//  main
//
//  Created by Daniel Cho on 5/30/26.
//

#pragma once

#include "engine/common/system.h"
#include "engine/graphics/graphics.h"

class RenderSystem : public System {
    
    Graphics* graphics;
    
public:
    RenderSystem(Graphics* graphics) : graphics(graphics) {}
    
    void init() override {}
    void update(float deltaTime) override {}
    void cleanup() override {}
    
private:
    
};
