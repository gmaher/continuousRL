class Config():
    gamma = 0.99
    clip_val = 10
    tau = 0.001
    grad_clip = True
    learn_frequency = 1
    num_episodes = 1000
    lr = 1e-3
    lr_mu = 1e-4
    render_frequency = 30
    max_action = 2.0
    l2reg = 0.01
    mu_reg = 0.000001
    update_frequency = 1
    noise_min = 0.0
    max_steps = 200
