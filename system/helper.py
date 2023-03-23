import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def compute_angle(vec_1, vec_2):
    length_vector_1 = np.linalg.norm(vec_1)
    length_vector_2 = np.linalg.norm(vec_2)
    if length_vector_1 == 0 or length_vector_2 == 0:
        return 0
    unit_vector_1 = vec_1 / length_vector_1
    unit_vector_2 = vec_2 / length_vector_2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    vec = np.cross([vec_1[0], vec_1[1], 0], [vec_2[0], vec_2[1], 0])

    return angle * np.sign(vec[2])


def compute_theta(vec):
    if vec[0] == 0:
        angle = np.pi/2
    else:
        angle = np.arctan(abs(vec[1] / vec[0]))
        if vec[0] < 0:
            angle = np.pi - angle
    return angle * np.sign(vec[1])


def compute_axis_limits(arena_size, xy_coordinates=None, environment=None):
    temp_arena_size = 1.1 * arena_size
    limits_t = [- temp_arena_size, temp_arena_size,
                - temp_arena_size, temp_arena_size]
    if environment == "linear_sunburst":
        limits_t = [0, arena_size,
                    0, arena_size]
    if xy_coordinates is not None:
        # Compute Axis limits for plot
        x, y = zip(*xy_coordinates)
        limits_t = [np.around(min(x), 1) - 0.1, np.around(max(x), 1) + 0.1,
                    np.around(min(y), 1) - 0.1, np.around(max(y), 1) + 0.1]

    x_t_width = limits_t[1] - limits_t[0]
    y_t_width = limits_t[3] - limits_t[2]

    x_width = 432.0
    y_width = 306.0
    ratio = x_width / y_width
    if ratio >= x_t_width / y_t_width:
        rescaled_width = y_t_width * ratio
        diff = (rescaled_width - x_t_width) / 2
        limits_t[0] = limits_t[0] - diff
        limits_t[1] = limits_t[1] + diff
    else:
        rescaled_width = x_t_width / ratio
        diff = (rescaled_width - y_t_width) / 2
        limits_t[2] = limits_t[2] - diff
        limits_t[3] = limits_t[3] + diff

    return limits_t


def Printsimulationtimeresult(t_total,t_episode,dt,GCnetTimes,
                              PCnetTimes,CognitivemapnetTimes,bcTimes,
                              robotTimes,goalsearchTimes,plotTimes,
                              hdcplotTimes,decodeTimes,timesteps_neuron,
                              plotfps,avs,r2d,errs,errs_signed,
                              errs_noisy_signed,use_noisy_av,thetas,rtplot_bc,rtplot_hdc):
    print("print the result using function")
    X = np.arange(0.0, t_episode, dt)
    print('length of X',len(X))
    cahv = [avs[i] - avs[i - 1] if i > 0 else avs[0] for i in range(len(avs))]
    cerr = [errs_signed[i] - errs_signed[i - 1] if i > 0 else errs_signed[0] for i in range(len(errs_signed))]
    corr, _ = pearsonr(cahv, cerr)
    # error noisy integration vs. noisy HDC
    if use_noisy_av:
        plt.plot(X, errs_noisy_signed, label="Noisy integration")
        plt.plot(X, errs_signed, label="Noisy HDC")
        plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
        plt.xlabel("time (s)")
        plt.ylabel("error (deg)")
        plt.legend()
        plt.show()
    print("\n\n\n")
    print("############### Begin Simulation results ###############")
    # performance tracking
    print("Total time (real): {:.2f} s, Total time (simulated): {:.2f} s, simulation speed: {:.2f}*RT".format(t_total,
                                                                                                              t_episode,
                                                                                                              t_episode / t_total))
    # print("Average step time network of HDC:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(hdcnetTimes),
    #                                                                        1.0 / np.mean(hdcnetTimes)))
    print("Average step time network of GC:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(GCnetTimes),
                                                                           1.0 / np.mean(GCnetTimes)))
    print("Average step time network of PC:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(PCnetTimes),
                                                                           1.0 / np.mean(PCnetTimes)))
    print("Average step time network of Cognitivemap:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(CognitivemapnetTimes),
                                                                           1.0 / np.mean(CognitivemapnetTimes)))
    print("Average step time network of BC:  {:.4f} ms; {} it/s possible".format(
        1000.0 * np.mean(bcTimes),
        1.0 / np.mean(bcTimes)))

    print("Average step time robot:    {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(robotTimes),
                                                                           1.0 / np.mean(robotTimes)))
    print("Average step time for searching goal:    {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(goalsearchTimes),
                                                                           1.0 / np.mean(goalsearchTimes)))
    if rtplot_bc:
        print("Average step time plotting for BC: {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(plotTimes),
                                                                               1.0 / np.mean(plotTimes)))
    if rtplot_hdc:
        print("Average step time plotting for HDC: {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(hdcplotTimes),
                                                                               1.0 / np.mean(hdcplotTimes)))
    time_coverage = 0.0
    print("Average time decoding:      {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(decodeTimes),
                                                                           1.0 / np.mean(decodeTimes)))
    # print("Steps done network of HDC:  {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X) * timesteps_neuron,
    #                                                                               len(X) * timesteps_neuron * np.mean(
    #                                                                                   hdcnetTimes), 100 * len(
    #         X) * timesteps_neuron * np.mean(hdcnetTimes) / t_total))
    # time_coverage += 100 * len(X) * timesteps_neuron * np.mean(hdcnetTimes) / t_total
    print("Steps done robot:    {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(robotTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      robotTimes) / t_total))
    time_coverage += 100 * len(X) * np.mean(robotTimes) / t_total
    '''
    if rtplot_hdc:
        print("Steps done HDCplotting: {}; Time: {:.3f} s; {:.2f}% of total time".format(int(t_episode / plotfps),
                                                                                      int(t_episode / plotfps) * np.mean(
                                                                                          plotTimes), 100 * int(
                t_episode / plotfps) * np.mean(plotTimes) / t_total))
        time_coverage += 100 * int(t_episode / plotfps) * np.mean(plotTimes) / t_total
    print("Steps done decoding: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(decodeTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      decodeTimes) / t_total))
    '''
    print("Steps done searching goal: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(goalsearchTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      goalsearchTimes) / t_total))
    print("Steps done GC network: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(GCnetTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      GCnetTimes) / t_total))
    print("Steps done PC network: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(PCnetTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      PCnetTimes) / t_total))
    print("Steps done Cognitive map : {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(CognitivemapnetTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      CognitivemapnetTimes) / t_total))
    if rtplot_bc:
        print("Steps done BC : {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(bcTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      bcTimes) / t_total))
    time_coverage += 100 * len(X) * np.mean(decodeTimes) / t_total
    print("Time covered by the listed operations: {:.3f}%".format(time_coverage))
    print("maximum angular velocity: {:.4f} deg/s".format(max(avs) * r2d))
    print("average angular velocity: {:.4f} deg/s".format(sum([r2d * (x / len(avs)) for x in avs])))
    print("median angular velocity:  {:.4f} deg/s".format(np.median(avs)))
    print("maximum error: {:.4f} deg".format(max(errs)))
    print("average error: {:.4f} deg".format(np.mean(errs)))
    print("median error:  {:.4f} deg".format(np.median(errs)))
    print("################ End Simulation results ################")
    print("\n\n\n")

    # close real-time plot
    # plt.close()
    # plt.ioff()

    # plot error and angular velocity
    fig, ax1 = plt.subplots()
    # ax1.set_xlim(200, 375)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("error (deg)")
    ax1.set_ylim(-13.5, 13.5)
    ax1.plot(X, errs_signed, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("angular velocity (deg/s)")
    ax2.set_ylim(-50, 50)
    ax2.plot(X, [x * r2d for x in avs], color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot only error
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    plt.ylim(-1.6, 1.6)
    plt.xlim(0.0, t_episode)
    plt.plot(X, errs_signed)
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot only angular velocity
    plt.xlabel("time (s)")
    plt.ylabel("angular velocity (deg/s)")
    plt.ylim(-50, 50)
    plt.xlim(0.0, t_episode)
    plt.plot(X, [x * r2d for x in avs])
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot total rotation
    totalMovements = [0.0] * len(thetas)
    for i in range(1, len(avs)):
        totalMovements[i] = totalMovements[i - 1] + abs(r2d * thetas[i - 1])
    plt.plot(X, totalMovements)
    plt.xlabel("time (s)")
    plt.ylabel("total rotation (deg)")
    plt.show()

    # plot relative error # only used for PI evaluation
    # begin after 20%
    begin_relerror = int(0.2 * len(X))
    plt.plot(X[begin_relerror:len(X)], [100 * errs[i] / totalMovements[i] for i in range(begin_relerror, len(errs))])
    plt.xlabel("time (s)")
    plt.ylabel("relative error (%)")
    plt.show()

    # plot change in angular velocity vs. change in error # only used for PI evaluation
    plt.scatter(cahv, cerr)
    plt.plot([min(cahv), max(cahv)], [corr * min(cahv), corr * max(cahv)],
             label="linear approximation with slope {:.2f}".format(corr), color="tab:red")
    plt.legend()
    plt.xlabel("change in angular velocity (deg/s)")
    plt.ylabel("change in error (deg)")
    plt.show()
