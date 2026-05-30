//
//  main.cpp
//  sim-1
//
//  Created by Daniel Cho on 5/29/26.
//

#include <iostream>

#include "application.h"

int main() {
    
    std::cout << "=== main ===" << std::endl;
    
    try {
        Application *app = new Application;
        app->init();
        app->start();
        app->cleanup();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
    
}


//#include "engine/physics/physics.h"


// initialize
    // camera + plane

// simple rendering pipeline with glfw

//    Graphics *graphics = new Graphics();
//
//    graphics->initWindow();
//    graphics->createInstance();
//    graphics->setupDebugMessenger();
//    graphics->createSurface();
//
//    graphics->pickPhysicalDevice();
//    graphics->createLogicalDevice();
//    graphics->createCommandPool();
//    graphics->createCommandBuffers();
//    graphics->createSyncObjects();
//
//    graphics->createTextureSampler();
//
//    graphics->loadQuad(); // temp
//
////    graphics.loadTexture("viking_room.png",
////                         modelTextureImage,
////                         modelTextureImageMemory,
////                         modelTextureImageView,
////                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT
////                         | VK_IMAGE_USAGE_TRANSFER_DST_BIT
////                         | VK_IMAGE_USAGE_SAMPLED_BIT,
////                         1);
////
////    graphics.transitionImageLayout(modelTextureImage,
////                                   VK_FORMAT_R8G8B8A8_SRGB,
////                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
////                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
////                                   1);
////
////    graphics.textureImages.push_back(modelTextureImage);
////    graphics.textureImageMemories.push_back(modelTextureImageMemory);
////    graphics.textureImageViews.push_back(modelTextureImageView);
////
////    // load canvas quad
////
////    auto model = Graphics::loadObj("viking_room.obj");
////    graphics.pushModel(model); // todo: which objs to draw, with which materials
//
//    graphics->createVertexBuffer();
//    graphics->createIndexBuffer();
//    graphics->createUniformBuffers();
//
//    graphics->createSwapChain();
//    graphics->createSwapChainImageViews();
//    graphics->createSwapChainRenderPass();
//    graphics->createSwapChainDescriptorSetLayout();
//    graphics->createSwapChainGraphicsPipeline(resolveBundlePath("shader_vert.spv"),
//                                              resolveBundlePath("shader_frag.spv"));
//    graphics->createSwapChainColorResources();
//    graphics->createSwapChainDepthResources();
//    graphics->createSwapChainFramebuffers();
//    graphics->createSwapChainDescriptorPool();



//void GraphicsApplication::draw() {
//    
//    uint32_t imageIndex;
//    
//    graphics.startFrame(imageIndex);
//    
//    static auto startTime = std::chrono::high_resolution_clock::now();
//    auto currentTime = std::chrono::high_resolution_clock::now();
//    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
//    
//    int windowWidth, windowHeight;
//    glfwGetWindowSize(graphics.window, &windowWidth, &windowHeight);
//
//    glm::mat4 view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
//                                 glm::vec3(0.0f, 0.0f, 0.0f),
//                                 glm::vec3(0.0f, 0.0f, 1.0f));
//    
//    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
//                                      windowWidth / (float) windowHeight,
//                                      0.1f,
//                                      10.0f);
//    proj[1][1] *= -1;
//    
//    graphics.updateGlobalUBO(view, proj);
//    
//    std::vector<InstanceSSBO> instances(config.graphicsConfig.MAX_ENTITIES);
//    instances[0].model = glm::rotate(glm::mat4(1.0f), time * glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
//
//    graphics.updateInstanceSSBOs(instances);
//    
//    graphics.submitFrame(imageIndex);
//
//}
//
//void GraphicsApplication::run() {
//    
//    // actual main loop
//    while (!glfwWindowShouldClose(graphics.window)) {
//        glfwPollEvents();
//        draw();
//    }
//    vkDeviceWaitIdle(graphics.device);
//    
//}
//
//void GraphicsApplication::cleanup() {
//    
//    graphics.cleanup();
//    
//}
