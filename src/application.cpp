//
//  application.c
//  main
//
//  Created by Daniel Cho on 5/30/26.
//

#include <iostream>
#include <chrono>

#include "application.h"
#include "systems/render_system.h"

void Application::init() {
    std::cout << "[application] init" << std::endl;
    
    // graphics object essentials
    
    graphics = new Graphics();
    graphics->initWindow();
            
    graphics->createInstance();
    graphics->setupDebugMessenger();
    graphics->createSurface();

    graphics->pickPhysicalDevice();
    graphics->createLogicalDevice();
    graphics->createCommandPool();
    graphics->createCommandBuffers();
    graphics->createSyncObjects();
    
    // systems
    
    systems.push_back(new RenderSystem(graphics));
    
    // scene loader for entity + components
    
    // system init functions
    
    for (System* system : systems) {
        system->init();
    }

}

void Application::start() {
    std::cout << "[application] start" << std::endl;
    
    auto lastTime = std::chrono::high_resolution_clock::now();
    
    while (!glfwWindowShouldClose(graphics->window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float, std::milli>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        glfwPollEvents();
        for (System* system : systems) {
            system->update(deltaTime);
        }
    }
}

void Application::cleanup() {
    std::cout << "[application] cleanup" << std::endl;
    
    for (System* system : systems) {
        system->cleanup();
    }
    
    graphics->cleanup();
    delete graphics;
}
