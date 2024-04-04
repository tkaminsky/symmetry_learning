import pygame

def manualController(env):
    env.reset()

    # Do manual control loop with WASD keys using pygame

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end_it = True
                pygame.quit()
                quit()
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            obs, reward, done, info = env.step(0)
            # print("reward: ", reward)
        if keys[pygame.K_d]:
            obs, reward, done, info = env.step(1)
            # print("reward: ", reward)       
        if keys[pygame.K_s]:
            obs, reward, done, info = env.step(2)
            # print("reward: ", reward)
        if keys[pygame.K_a]:
            obs, reward, done, info = env.step(3)
            # print("reward: ", reward)
        if keys[pygame.K_e]:
            obs, reward, done, info = env.step(4)
            # print("reward: ", reward)
        if keys[pygame.K_q]:
            obs, reward, done, info = env.step(5)
            # print("reward: ", reward)

        if done:
            print("DONE")
            print(reward)
            obs, _ = env.reset()
            done = False

        env.render()

        env.clock.tick(60)