import matplotlib.pyplot as plt
import numpy as np

TUM_colors = {
                'TUMBlue': '#0065BD',
                'TUMSecondaryBlue': '#005293',
                'TUMSecondaryBlue2': '#003359',
                'TUMBlack': '#000000',
                'TUMWhite': '#FFFFFF',
                'TUMDarkGray': '#333333',
                'TUMGray': '#808080',
                'TUMLightGray': '#CCCCC6',
                'TUMAccentGray': '#DAD7CB',
                'TUMAccentOrange': '#E37222',
                'TUMAccentGreen': '#A2AD00',
                'TUMAccentLightBlue': '#98C6EA',
                'TUMAccentBlue': '#64A0C8'
}


def add_environment(ax, env):
    if env.env_model == "obstacles":
        box = plt.Rectangle((-1.75, -0.5), 0.5, 1, color='k')
        ax.add_artist(box)
        circle = plt.Circle((1, -1), 0.4, color='k')
        ax.add_artist(circle)
        circle = plt.Circle((1, 1), 0.3, color='k')
        ax.add_artist(circle)
        circle = plt.Circle((-0.8, 2), 0.3, color='k')
        ax.add_artist(circle)
    elif env.env_model == "linear_sunburst":
        if env.door_option == "plane_doors_open35":
            doors = [1, 3, 7]  # [1, 3, 5, 7]
        elif env.door_option == "plane_doors_open15":
            doors = [3, 5, 7]
        elif env.door_option == "plane_doors_open1":
            doors = [3, 5, 7, 9]
        elif env.door_option == "plane_doors_allopen":
            doors = []
        else:
            doors = [1, 3, 5, 7]  # [1, 3, 5, 7]
        # doors = [1, 3, 5, 7]  # [1, 3, 5, 7]
        for x in doors:
            plot_box = plt.Rectangle((x, 5.4), 1, 0.2, color=TUM_colors['TUMGray'])
            ax.add_artist(plot_box)
        boxes = [0, 2, 4, 6, 8, 10]
        for x in boxes:
            plot_box = plt.Rectangle((x, 5), 1, 2, color=TUM_colors['TUMLightGray'])
            ax.add_artist(plot_box)
        boxes = [1, 3, 5, 7, 9]
        for x in boxes:
            plot_box = plt.Rectangle((x, 8), 1, 1, color=TUM_colors['TUMLightGray'])
            ax.add_artist(plot_box)
        plot_box = plt.Rectangle((-0.1, -0.1), 11.2, 11.2, color=TUM_colors['TUMLightGray'], fc='none',
                                 ec=TUM_colors['TUMLightGray'], linewidth=5)
        ax.add_artist(plot_box)


def add_robot(ax, env):
    xy = env.xy_coordinates[-1]
    circle = plt.Circle((xy[0], xy[1]), 0.2, color=TUM_colors['TUMDarkGray'], alpha=1)
    ax.add_artist(circle)

    angle = env.orientation_angle[-1]
    ax.quiver(xy[0], xy[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4, color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)
