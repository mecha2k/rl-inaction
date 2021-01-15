import gym

env = gym.make("CartPole-v0")
for episode in range(20):
    observation = env.reset()
    for n in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f"episode finished after {n+1} timesteps.")
            break


# from pyglet.gl import *
#
# # Direct OpenGL commands to this window.
# window = pyglet.window.Window()
#
#
# @window.event
# def on_draw():
#     glClear(GL_COLOR_BUFFER_BIT)
#     glLoadIdentity()
#     glBegin(GL_TRIANGLES)
#     glVertex2f(0, 0)
#     glVertex2f(window.width, 0)
#     glVertex2f(window.width, window.height)
#     glEnd()
#
#
# pyglet.app.run()
