class Config():
    gamma = 0.99
    clip_val = 10
    tau = 0.001
    grad_clip = True
    learn_frequency = 1
    num_episodes = 1000
    lr = 1e-3
    lr_mu = 1e-4
    render_frequency = 1
    max_action = 2.0
    l2reg = 1e-6
    mu_reg = 0.000001
    update_frequency = 1
    noise_min = 0.0
    max_steps = 500
    plot_frequency = 5
    start_train = 1
    train_iterations = 1

    model_name = 'model'
    plots_dir = 'plots'
